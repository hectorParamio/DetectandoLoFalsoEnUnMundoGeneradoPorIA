import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from albumentations import Compose, Normalize, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
import face_alignment
from tqdm import tqdm

# Load the CNN architecture from the existing model
class FaceClassifierCNN(nn.Module):
    def __init__(self):
        super(FaceClassifierCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Calculate output size for fully connected layers
        # For 128x128 images, after 2 max pooling layers = 32x32
        fc_input_size = 64 * 32 * 32
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # 2 classes: real or AI-generated
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# Initialize face alignment model
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

# Face cache for processed images
face_cache = {}
no_landmarks = 0

# Function to align faces
def align_face(image_path):
    global no_landmarks
    
    # Check if image is already in cache
    if image_path in face_cache:
        return face_cache[image_path]
    
    try:
        # Read image
        print(f"Reading image: {os.path.basename(image_path)}", end='\r')
        img = cv2.imread(image_path)
        if img is None:
            print(f"\nWarning: Could not read image file: {image_path}")
            return None

        # Convert to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to a smaller size for faster processing while maintaining aspect ratio
        height, width = img_rgb.shape[:2]
        max_dim = 400  # Maximum size for processing
        
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img_small = cv2.resize(img_rgb, (int(width * scale), int(height * scale)))
        else:
            img_small = img_rgb
            
        # Faster face detection using Haar Cascade from OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # If no face is detected, use the entire image
            no_landmarks += 1
            result = cv2.resize(img_rgb, (128, 128))
            face_cache[image_path] = result
            return result
        
        # Get the first detected face
        x, y, w, h = faces[0]
        
        # Adjust coordinates to the original image
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
        face_img = cv2.resize(face_img, (128, 128))
        
        # Save to cache
        face_cache[image_path] = face_img
        return face_img
        
    except Exception as e:
        print(f"\nError processing {image_path}: {e}")
        # In case of error, return the resized image if possible
        try:
            result = cv2.resize(img_rgb, (128, 128))
            face_cache[image_path] = result
            return result
        except:
            return None

# Dataset class 
class FaceDataset(Dataset):
    def __init__(self, images_dir, label, transform=None):
        self.image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
        self.label = label  # 0 for real, 1 for fake
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Use the same face alignment as in training
        image = align_face(img_path)
        
        if image is None:
            # Handle corrupted images by returning a black image
            print(f"Warning: Could not process image {img_path}")
            image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, self.label, img_path

# Define transformations for data augmentation
train_transforms = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.2),
    ShiftScaleRotate(p=0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transforms = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def prepare_datasets(real_dir, fake_dir, test_size=0.2, val_size=0.2):
    # Get all real and fake image paths
    real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
    fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create labels (0 for real, 1 for fake)
    real_labels = [0] * len(real_images)
    fake_labels = [1] * len(fake_images)
    
    # Combine datasets
    all_images = real_images + fake_images
    all_labels = real_labels + fake_labels
    
    # Split into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, all_labels, test_size=test_size, random_state=42, stratify=all_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )
    
    # Create custom datasets for each split
    train_dataset = []
    for img, label in zip(X_train, y_train):
        train_dataset.append((img, label))
    
    val_dataset = []
    for img, label in zip(X_val, y_val):
        val_dataset.append((img, label))
    
    test_dataset = []
    for img, label in zip(X_test, y_test):
        test_dataset.append((img, label))
    
    return train_dataset, val_dataset, test_dataset

# Custom dataset class for the prepared data
class PreparedDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # List of (image_path, label) tuples
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        # Align face
        image = align_face(img_path)
        
        if image is None:
            # Handle corrupted images
            print(f"Warning: Could not process image {img_path}")
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label, img_path

def analyze_model_outputs(model, dataloader, device, save_path=None):
    """Analyze the raw output logits of the model before classification decision"""
    print("Starting model output analysis...")
    model.eval()
    outputs_class0 = []  # Logits for class 0 (real)
    outputs_class1 = []  # Logits for class 1 (fake)
    true_labels = []
    
    print(f"Processing {len(dataloader)} batches of data...")
    with torch.no_grad():
        for batch_idx, (inputs, labels, _) in enumerate(dataloader):
            if batch_idx % 5 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}...")
            
            inputs = inputs.to(device)
            
            # Get raw logits from model
            outputs = model(inputs)
            
            # Store outputs by true class
            for i, label in enumerate(labels):
                if label.item() == 0:  # Real
                    outputs_class0.append(outputs[i].cpu().numpy())
                else:  # Fake
                    outputs_class1.append(outputs[i].cpu().numpy())
                    
                true_labels.append(label.item())
    
    print("Converting to numpy arrays...")
    # Convert to numpy arrays
    outputs_class0 = np.array(outputs_class0)
    outputs_class1 = np.array(outputs_class1)
    
    print(f"Collected {len(outputs_class0)} real samples and {len(outputs_class1)} fake samples")
    
    print("Creating plots...")
    # Plot histograms of logits
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(outputs_class0[:, 0], bins=30, alpha=0.5, label='Real Samples - Logit for Real Class')
    plt.hist(outputs_class1[:, 0], bins=30, alpha=0.5, label='Fake Samples - Logit for Real Class')
    plt.xlabel('Logit Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Logits for Real Class (0)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(outputs_class0[:, 1], bins=30, alpha=0.5, label='Real Samples - Logit for Fake Class')
    plt.hist(outputs_class1[:, 1], bins=30, alpha=0.5, label='Fake Samples - Logit for Fake Class')
    plt.xlabel('Logit Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Logits for Fake Class (1)')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot to the specified path or default
    if save_path:
        print(f"Saving plot to {save_path}...")
        plt.savefig(save_path)
    else:
        print("Saving plot to default location...")
        plt.savefig('model_logits_distribution.png')
    plt.close()
    
    print("Calculating statistics...")
    # Return statistics
    return {
        'real_mean_logit0': outputs_class0[:, 0].mean(),
        'real_mean_logit1': outputs_class0[:, 1].mean(),
        'fake_mean_logit0': outputs_class1[:, 0].mean(),
        'fake_mean_logit1': outputs_class1[:, 1].mean(),
        'real_std_logit0': outputs_class0[:, 0].std(),
        'real_std_logit1': outputs_class0[:, 1].std(),
        'fake_std_logit0': outputs_class1[:, 0].std(),
        'fake_std_logit1': outputs_class1[:, 1].std()
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=5, device='cpu', results_dir=None):
    """Train the model with validation"""
    model = model.to(device)
    
    # Track best validation accuracy
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Save statistics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, 'ours_finetunned.pth'))
            print(f'Model saved with validation accuracy: {val_acc:.4f}')
    
    return model, history

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, _ in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Handle cases where precision/recall might be undefined
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    # Always include both classes in confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    return accuracy, precision, recall, f1, conf_matrix

def plot_training_history(history, save_path=None):
    print("Creating training history plots...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot to the specified path or default
    if save_path:
        print(f"Saving training history plot to {save_path}")
        plt.savefig(save_path)
    else:
        print("Saving training history plot to default location")
        plt.savefig('training_history.png')
    
    plt.close()
    print("Training history plots created and saved")

def plot_confusion_matrix(conf_matrix, title="Confusion Matrix", save_path=None):
    print(f"Creating confusion matrix plot with title: {title}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'AI-Generated'],
                yticklabels=['Real', 'AI-Generated'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    
    # Save the plot to the specified path or default
    if save_path:
        print(f"Saving confusion matrix to {save_path}")
        plt.savefig(save_path)
    else:
        filename = f"{title.lower().replace(' ', '_')}.png"
        print(f"Saving confusion matrix to {filename}")
        plt.savefig(filename)
    
    plt.close()
    print("Confusion matrix plot created and saved")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    results_dir = "ourmodel_finetunned_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    
    # Dataset paths
    real_dir = "datasets/image/AI-Face-Detection/real"
    fake_dir = "datasets/image/AI-Face-Detection/fake"
    
    print(f"Loading images from: {real_dir} and {fake_dir}")
    
    # Step 1: Load the pre-trained model and analyze its outputs
    print("\n1. Loading pre-trained model and analyzing outputs...")
    
    # Create datasets for analysis
    print("Creating datasets for analysis...")
    analysis_real_dataset = FaceDataset(real_dir, label=0, transform=val_transforms)
    print(f"Created real dataset with {len(analysis_real_dataset)} images")
    analysis_fake_dataset = FaceDataset(fake_dir, label=1, transform=val_transforms)
    print(f"Created fake dataset with {len(analysis_fake_dataset)} images")
    
    # Create combined dataset for analysis
    print("Creating combined dataset...")
    analysis_combined_dataset = ConcatDataset([analysis_real_dataset, analysis_fake_dataset])
    print(f"Combined dataset has {len(analysis_combined_dataset)} images")
    print("Creating data loader...")
    analysis_loader = DataLoader(analysis_combined_dataset, batch_size=16, shuffle=True)
    
    # Load pre-trained model
    print(f"Loading pre-trained model from 'ourmodel_results/ours.pth'...")
    try:
        pretrained_model = FaceClassifierCNN()
        pretrained_model.load_state_dict(torch.load('ourmodel_results/ours.pth', map_location=device))
        pretrained_model.to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Analyze model outputs
    print("Starting model analysis...")
    model_stats = analyze_model_outputs(
        pretrained_model, 
        analysis_loader, 
        device,
        save_path=os.path.join(results_dir, 'model_logits_distribution.png')
    )
    print("Model output statistics:")
    for key, value in model_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Step 2: Prepare data for fine-tuning
    print("\n2. Preparing data for fine-tuning...")
    train_data, val_data, test_data = prepare_datasets(real_dir, fake_dir)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create datasets with appropriate transforms
    train_dataset = PreparedDataset(train_data, transform=train_transforms)
    val_dataset = PreparedDataset(val_data, transform=val_transforms)
    test_dataset = PreparedDataset(test_data, transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Step 3: Fine-tune the model
    print("\n3. Fine-tuning the model...")
    
    # Initialize model with pre-trained weights
    model = FaceClassifierCNN()
    model.load_state_dict(torch.load('ourmodel_results/ours.pth', map_location=device))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for fine-tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Fine-tune the model
    finetuned_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        scheduler=scheduler, num_epochs=10, device=device, results_dir=results_dir
    )
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(results_dir, 'training_history.png'))
    
    # Step 4: Evaluate the fine-tuned model
    print("\n4. Evaluating the fine-tuned model...")
    
    # Load the best fine-tuned model
    best_model = FaceClassifierCNN()
    best_model.load_state_dict(torch.load(os.path.join(results_dir, 'ours_finetunned.pth'), map_location=device))
    best_model.to(device)
    
    # Evaluate on test set
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(best_model, test_loader, device)
    
    print("\n----- Test Results -----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, "Test Confusion Matrix", 
                         save_path=os.path.join(results_dir, 'confusion_matrix.png'))
    
    # Save metrics to text file
    metrics_path = os.path.join(results_dir, 'model_metrics.txt')
    print(f"Saving metrics to {metrics_path}")
    with open(metrics_path, 'w') as f:
        f.write("=== Fine-tuned CNN Model Performance ===\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        
        # Calculate per-class accuracy
        real_correct = conf_matrix[0, 0]
        real_total = conf_matrix[0, 0] + conf_matrix[0, 1]
        real_acc = real_correct / real_total if real_total > 0 else 0
        
        fake_correct = conf_matrix[1, 1]
        fake_total = conf_matrix[1, 0] + conf_matrix[1, 1]
        fake_acc = fake_correct / fake_total if fake_total > 0 else 0
        
        f.write(f"Real images accuracy: {real_acc:.4f}\n")
        f.write(f"AI-generated images accuracy: {fake_acc:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("                 Predicted\n")
        f.write("                 Real    AI-gen\n")
        f.write(f"Actual Real     {conf_matrix[0][0]:<7} {conf_matrix[0][1]}\n")
        f.write(f"       AI-gen   {conf_matrix[1][0]:<7} {conf_matrix[1][1]}\n\n")
        
        # Model information
        f.write("=== Model Information ===\n\n")
        f.write("Architecture: 2-layer CNN with fully connected layers\n")
        total_params = sum(p.numel() for p in best_model.parameters())
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Images with no landmarks detected: {no_landmarks}\n")
    
    print(f"Metrics saved to: {metrics_path}")
    
    # Print number of images with no landmarks
    print(f"\nStatistics:")
    print(f"Images with no landmarks detected: {no_landmarks}")
    
    print(f"\nFine-tuning completed! The model has been saved as '{os.path.join(results_dir, 'ours_finetunned.pth')}'")

if __name__ == "__main__":
    main() 