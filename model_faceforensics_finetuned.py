import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import pandas as pd
import random
from tqdm import tqdm
import warnings
from torch.utils.model_zoo import load_url
from sklearn.model_selection import train_test_split
from albumentations import Compose, Normalize, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, GaussNoise
from albumentations.pytorch import ToTensorV2
import time

# OPTIMIZATION SETTINGS
USE_SMALLER_IMAGES = False      # Reduce image size from 224 to 160 pixels
FACE_SIZE = 160 if USE_SMALLER_IMAGES else 224
BATCH_SIZE = 8                # Increased from 8 to 16
NUM_WORKERS = 4                # Use multiple workers for data loading
USE_AMP = True                 # Use automatic mixed precision
FREEZE_BACKBONE = True         # Freeze most of the model for faster training
PRE_PROCESS_FACES = True       # Pre-process and save all faces before training
MAX_IMAGES_PER_CLASS = 1000     # Limit total number of images for faster training (set to -1 for all)

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the models directory to the path so imports work
sys.path.append(os.path.abspath("."))

# Import required FaceForensics modules
from models.faceforensics_image.blazeface import FaceExtractor, BlazeFace
from models.faceforensics_image.architectures import fornet, weights

# Define paths to the dataset
DATASET_PATH = "datasets/image/AI-Face-Detection"
REAL_PATH = os.path.join(DATASET_PATH, "real")
FAKE_PATH = os.path.join(DATASET_PATH, "fake")

# Cache for processed faces
face_cache = {}

# Global variables for model
faceforensics_model = None
face_extractor = None
device = None

# Add this line at the beginning of your script to suppress the update message
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Custom transformer for FaceForensics - using modern albumentations transformations
def get_transformer(face_policy, face_size, normalize=True, train=True):
    # Define normalization parameters (same as in FaceForensics)
    norm_mean = [0.485, 0.456, 0.406]  # ImageNet mean
    norm_std = [0.229, 0.224, 0.225]   # ImageNet std
    
    # Training transformations with augmentation
    if train:
        return Compose([
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            ShiftScaleRotate(p=0.2, shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
            GaussNoise(p=0.2),  # Modern replacement for IAAAdditiveGaussianNoise
            Normalize(mean=norm_mean, std=norm_std) if normalize else lambda x: x,
            ToTensorV2(),
        ])
    # Validation/testing transformations (just normalization)
    else:
        return Compose([
            Normalize(mean=norm_mean, std=norm_std) if normalize else lambda x: x,
            ToTensorV2(),
        ])

def load_faceforensics_model(training_mode=False):
    """
    Load FaceForensics model and set it up for training or evaluation
    
    Args:
        training_mode: If True, set model to training mode, otherwise evaluation mode
    """
    global faceforensics_model, face_extractor, device
    
    # Choose model architecture and training dataset
    net_model = 'EfficientNetAutoAttB4'
    train_db = 'DFDC'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device('cpu'))
    face_policy = 'scale'
    face_size = 224
    
    print("Loading FaceForensics model...")
    
    # Set up model
    model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
    faceforensics_model = getattr(fornet, net_model)().to(device)
    faceforensics_model.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
    
    if training_mode:
        faceforensics_model.train()
        print("Model set to training mode for fine-tuning")
    else:
        faceforensics_model.eval()
    
    print("Loaded pretrained weights for", net_model)
    
    # Face detection
    facedet = BlazeFace().to(device)
    facedet.load_weights("models/faceforensics_image/blazeface/blazeface.pth")
    facedet.load_anchors("models/faceforensics_image/blazeface/anchors.npy")
    face_extractor = FaceExtractor(facedet=facedet)
    print("Face detector loaded")
    
    return faceforensics_model

def align_face(image_path):
    """Extract and align face from an image using OpenCV Haar Cascade."""
    # Check if image is already in cache
    if image_path in face_cache:
        return face_cache[image_path]
    
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing while maintaining aspect ratio
        height, width = img_rgb.shape[:2]
        max_dim = 400
        
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img_small = cv2.resize(img_rgb, (int(width * scale), int(height * scale)))
        else:
            img_small = img_rgb
            
        # Use OpenCV Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # If no face detected, use entire image
            result = cv2.resize(img_rgb, (FACE_SIZE, FACE_SIZE))
            face_cache[image_path] = result
            return result
        
        # Get first detected face
        x, y, w, h = faces[0]
        
        # Adjust coordinates to original image if resized
        if max(height, width) > max_dim:
            inv_scale = 1.0 / scale
            x, y, w, h = int(x * inv_scale), int(y * inv_scale), int(w * inv_scale), int(h * inv_scale)
        
        # Add padding
        padding_factor = 0.3
        padding_x = int(w * padding_factor)
        padding_y = int(h * padding_factor)
        
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(width, x + w + padding_x)
        y2 = min(height, y + h + padding_y)
        
        # Crop and resize
        face_img = img_rgb[y1:y2, x1:x2]
        face_img = cv2.resize(face_img, (FACE_SIZE, FACE_SIZE))
        
        # Save to cache
        face_cache[image_path] = face_img
        return face_img
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        try:
            # If error occurs, try to return resized original
            result = cv2.resize(img_rgb, (FACE_SIZE, FACE_SIZE))
            face_cache[image_path] = result
            return result
        except:
            return None

# Custom dataset class for fine-tuning
class FaceForensicsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Align face
        aligned_face = align_face(img_path)
        
        if aligned_face is None:
            print(f"Warning: Could not process image {img_path}")
            aligned_face = np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)
        
        # Apply transformations if provided
        if self.transform:
            try:
                augmented = self.transform(image=aligned_face)
                aligned_face = augmented['image']
            except Exception as e:
                print(f"Error applying transformations to {img_path}: {str(e)}")
                if isinstance(self.transform, ToTensorV2):
                    aligned_face = torch.zeros((3, FACE_SIZE, FACE_SIZE))
                
        return aligned_face, self.labels[idx]

def prepare_datasets(test_size=0.2, val_size=0.2):
    """Prepare datasets for training, validation, and testing."""
    # Get all real and fake image paths
    real_images = [os.path.join(REAL_PATH, img) for img in os.listdir(REAL_PATH) 
                  if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_images = [os.path.join(FAKE_PATH, img) for img in os.listdir(FAKE_PATH) 
                  if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limit dataset size for faster training if requested
    if MAX_IMAGES_PER_CLASS > 0:
        if len(real_images) > MAX_IMAGES_PER_CLASS:
            real_images = random.sample(real_images, MAX_IMAGES_PER_CLASS)
        if len(fake_images) > MAX_IMAGES_PER_CLASS:
            fake_images = random.sample(fake_images, MAX_IMAGES_PER_CLASS)
    
    print(f"Using {len(real_images)} real images and {len(fake_images)} fake images")
    
    # Create labels: 0 for fake, 1 for real
    # Note: FaceForensics treats 0 as real, 1 as fake internally, but we need to invert for our loss function
    real_labels = [1] * len(real_images)
    fake_labels = [0] * len(fake_images)
    
    # Combine datasets
    all_images = real_images + fake_images
    all_labels = real_labels + fake_labels
    
    # Split data into train, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, all_labels, test_size=test_size, random_state=42, stratify=all_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )
    
    print(f"Training: {len(X_train)} images, Validation: {len(X_val)} images, Test: {len(X_test)} images")
    
    # Pre-process all faces if enabled (significantly speeds up training)
    if PRE_PROCESS_FACES:
        print("Pre-processing faces for faster training...")
        all_paths = X_train + X_val + X_test
        
        for img_path in tqdm(all_paths, desc="Processing faces"):
            align_face(img_path)
        
        print(f"Processed {len(all_paths)} images")
    
    # Move this here and remove it from prepare_datasets
    pos_weight = torch.tensor([2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def fine_tune_model():
    """Fine-tune the FaceForensics model on the dataset."""
    global faceforensics_model, device
    
    start_time = time.time()
    
    # Create results directory
    results_dir = "faceforensics_finetunned_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model in training mode
    model = load_faceforensics_model(training_mode=True)
    
    # Freeze backbone layers for faster training if enabled
    if FREEZE_BACKBONE:
        print("Freezing backbone layers for faster training...")
        # Count total parameters before freezing
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Freeze early layers
        for name, param in model.named_parameters():
            if 'backbone.0' in name or 'backbone.1' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Count trainable parameters after freezing
        trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Reduced trainable parameters from {trainable_params:,} to {trainable_params_after:,} " +
              f"({trainable_params_after/total_params*100:.1f}% of total)")
    
    # Create custom transformers using our function
    face_policy = 'scale'
    face_size = FACE_SIZE
    normalizer = model.get_normalizer()
    
    # Data augmentation for training
    train_transformer = get_transformer(face_policy, face_size, normalize=True, train=True)
    
    # Simple normalization for validation and testing
    val_transformer = get_transformer(face_policy, face_size, normalize=True, train=False)
    
    # Prepare datasets
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_datasets()
    
    # Move this here and remove it from prepare_datasets
    pos_weight = torch.tensor([2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Create datasets
    train_dataset = FaceForensicsDataset(X_train, y_train, transform=train_transformer)
    val_dataset = FaceForensicsDataset(X_val, y_val, transform=val_transformer)
    test_dataset = FaceForensicsDataset(X_test, y_test, transform=val_transformer)
    
    # Create data loaders with multiple workers for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    
    # Use different learning rates for different parts of the model
    # Lower learning rate for pre-trained layers
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only update parameters that require gradients
            if 'backbone' in name:
                params_to_update.append({'params': param, 'lr': 0.00001})
            else:
                params_to_update.append({'params': param, 'lr': 0.0001})
    
    # Optimizer with weight decay
    optimizer = optim.Adam(params_to_update, weight_decay=0.0001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)
    
    # Initialize mixed precision training if enabled
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and torch.cuda.is_available() else None
    
    # Training loop
    num_epochs = 10
    best_val_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_size = 0
        
        # Wrap the training loop with tqdm for a progress bar
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    # Fix dimension mismatch by handling both single and batch cases
                    if outputs.dim() == 0:  # scalar output
                        outputs = outputs.unsqueeze(0)  # add batch dimension
                    loss = criterion(outputs.view(-1), labels.view(-1))
                    
                # Backward pass and optimize with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(inputs)
                # Fix dimension mismatch by handling both single and batch cases
                if outputs.dim() == 0:  # scalar output
                    outputs = outputs.unsqueeze(0)  # add batch dimension
                loss = criterion(outputs.view(-1), labels.view(-1))
                loss.backward()
                optimizer.step()
            
            # Convert from logits to probabilities
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.squeeze() == labels.data)
            processed_size += inputs.size(0)
        
        epoch_loss = running_loss / processed_size
        epoch_acc = running_corrects.double() / processed_size
        
        print(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.cpu().item())
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        processed_size = 0
        
        for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            # Forward pass (no gradient)
            with torch.no_grad():
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        # Fix dimension mismatch by handling both single and batch cases
                        if outputs.dim() == 0:  # scalar output
                            outputs = outputs.unsqueeze(0)  # add batch dimension
                        loss = criterion(outputs.view(-1), labels.view(-1))
                else:
                    outputs = model(inputs)
                    # Fix dimension mismatch by handling both single and batch cases
                    if outputs.dim() == 0:  # scalar output
                        outputs = outputs.unsqueeze(0)  # add batch dimension
                    loss = criterion(outputs.view(-1), labels.view(-1))
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.squeeze() == labels.data)
            processed_size += inputs.size(0)
        
        epoch_loss = running_loss / processed_size
        epoch_acc = running_corrects.double() / processed_size
        
        print(f"Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.cpu().item())
        
        # Update learning rate
        scheduler.step(epoch_loss)
        
        # Save best model
        if epoch_acc > best_val_accuracy:
            best_val_accuracy = epoch_acc
            model_path = os.path.join(results_dir, 'faceforensics_finetuned.pth')
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with validation accuracy: {epoch_acc:.4f}")
        
        # Print epoch timing information
        epoch_end = time.time()
        print(f"Epoch completed in {epoch_end - epoch_start:.2f} seconds")
    
    # Plot training history
    plot_training_history(history, results_dir)
    
    # Load best model for testing
    model_path = os.path.join(results_dir, 'faceforensics_finetuned.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Test the model
    print("\nEvaluating fine-tuned model on test set...")
    test_results = evaluate_model(model, test_loader, results_dir)
    
    # Print total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s"
    print(training_time_str)
    
    # Save metrics to a text file
    metrics_path = os.path.join(results_dir, 'model_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=== FaceForensics Fine-tuned Model Performance ===\n\n")
        f.write(f"Accuracy: {test_results[0]:.4f}\n")
        f.write(f"Precision: {test_results[1]:.4f}\n")
        f.write(f"Recall: {test_results[2]:.4f}\n")
        f.write(f"F1 Score: {test_results[3]:.4f}\n")
        f.write(f"Real images accuracy: {test_results[4]:.4f}\n")
        f.write(f"Fake images accuracy: {test_results[5]:.4f}\n\n")
        
        # Save training parameters
        f.write("=== Training Parameters ===\n\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Images per class: {MAX_IMAGES_PER_CLASS}\n")
        f.write(f"Image size: {FACE_SIZE}x{FACE_SIZE}\n")
        f.write(f"Backbone frozen: {FREEZE_BACKBONE}\n\n")
        
        # Save training time
        f.write(f"{training_time_str}\n")
        
        # Add dataset information
        f.write(f"\n=== Dataset Information ===\n\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
    
    print(f"Metrics saved to: {metrics_path}")
    
    return test_results

def evaluate_model(model, test_loader, results_dir):
    """Evaluate the model on a dataset."""
    model.eval()
    y_true = []
    y_pred = []
    y_pred_raw = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_pred_raw.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Adjust for potential class imbalance
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate class-specific accuracy
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    real_indices = (y_true_array == 1)
    fake_indices = (y_true_array == 0)
    
    real_acc = accuracy_score(y_true_array[real_indices], y_pred_array[real_indices]) if np.any(real_indices) else 0
    fake_acc = accuracy_score(y_true_array[fake_indices], y_pred_array[fake_indices]) if np.any(fake_indices) else 0
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print("\n=== Fine-tuned Model Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Real images accuracy: {real_acc:.4f}")
    print(f"Fake images accuracy: {fake_acc:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Fine-tuned Model')
    plt.savefig(os.path.join(results_dir, 'finetuned_confusion_matrix.png'))
    plt.close()
    
    # Analyze distribution of predictions
    plt.figure(figsize=(10, 6))
    plt.hist([proba for proba, label in zip(y_pred_raw, y_true) if label == 1], 
             bins=20, alpha=0.5, label='Real Images')
    plt.hist([proba for proba, label in zip(y_pred_raw, y_true) if label == 0], 
             bins=20, alpha=0.5, label='Fake Images')
    plt.xlabel('Prediction Probability (Real)')
    plt.ylabel('Count')
    plt.title('Distribution of Predictions - Fine-tuned Model')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'finetuned_predictions_distribution.png'))
    plt.close()
    
    return accuracy, precision, recall, f1, real_acc, fake_acc, cm

def plot_training_history(history, results_dir):
    """Plot the training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'finetuning_history.png'))
    plt.close()

def predict_with_finetuned_model(img_path):
    """
    Make a prediction using the fine-tuned model.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Probability that the image is real (0-1)
    """
    # Load the fine-tuned model
    model = load_faceforensics_model(training_mode=False)
    results_dir = "faceforensics_finetunned_results"
    model_path = os.path.join(results_dir, 'faceforensics_finetuned.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Process the image
    aligned_face = align_face(img_path)
    if aligned_face is None:
        print(f"Failed to process image: {img_path}")
        return 0.5
    
    # Create transformer
    transformer = get_transformer('scale', 224, normalize=True, train=False)
    
    # Convert to tensor
    image_tensor = transformer(image=aligned_face)['image'].unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
    
    # Return the probability (FaceForensics model: 0=real, 1=fake)
    # We inverted the labels during training, so here we need to invert back
    return 1 - prob

if __name__ == "__main__":
    print("Starting model fine-tuning process...")
    
    # Always fine-tune from the pre-trained weights
    print("Fine-tuning the model from pre-trained weights...")
    
    # Run fine-tuning to improve the model
    results = fine_tune_model()
    
    print("\nFine-tuning complete!")
    print(f"Final model metrics: Accuracy={results[0]:.4f}, Precision={results[1]:.4f}, Recall={results[2]:.4f}, F1={results[3]:.4f}")
    print(f"Class accuracies: Real={results[4]:.4f}, Fake={results[5]:.4f}") 