import kagglehub
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import face_alignment
from albumentations import (
    Compose, RandomBrightnessContrast, HorizontalFlip,
    Normalize, ShiftScaleRotate, ColorJitter
)
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random

"""# Code"""

download_path = kagglehub.dataset_download("kaustubhdhote/human-faces-dataset")
real_path = os.path.join(download_path, "Human Faces Dataset", "Real Images")
ai_path = os.path.join(download_path, "Human Faces Dataset", "AI-Generated Images")

# Counters for statistics
no_landmarks = 0

# Initialize face alignment model
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

# Implementar caché para imágenes procesadas
face_cache = {}

# Function to align faces - versión optimizada
def align_face(image_path):
    global no_landmarks
    
    # Check if the image is already in cache
    if image_path in face_cache:
        return face_cache[image_path]
    
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image file: {image_path}")
            return None

        # Convert to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Simple resize if the image is too small
        height, width = img_rgb.shape[:2]
        if height < 32 or width < 32:
            print(f"Warning: Image too small: {image_path}, {width}x{height}")
            result = cv2.resize(img_rgb, (128, 128))
            face_cache[image_path] = result
            return result
        
        # Resize to a manageable size for faster processing
        # while maintaining aspect ratio
        max_dim = 400  # Max dimension for processing
        
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img_small = cv2.resize(img_rgb, (int(width * scale), int(height * scale)))
        else:
            img_small = img_rgb
            
        # Use OpenCV's Haar Cascade for fast face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # If no face detected, use the entire image
            no_landmarks += 1
            result = cv2.resize(img_rgb, (128, 128))
            face_cache[image_path] = result
            return result
        
        # Get first detected face
        x, y, w, h = faces[0]
        
        # Adjust coordinates to the original image if resized
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
        
        # Sanity check for valid crop coordinates
        if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            print(f"Warning: Invalid crop coordinates for {image_path}")
            result = cv2.resize(img_rgb, (128, 128))
            face_cache[image_path] = result
            return result
        
        # Crop and resize
        try:
            face_img = img_rgb[y1:y2, x1:x2]
            face_img = cv2.resize(face_img, (128, 128))
            
            # Save to cache
            face_cache[image_path] = face_img
            return face_img
        except Exception as e:
            print(f"Error cropping face from {image_path}: {e}")
            # Fall back to resizing the whole image
            result = cv2.resize(img_rgb, (128, 128))
            face_cache[image_path] = result
            return result
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        # In case of error, try to return a resized version of the original image
        try:
            if 'img_rgb' in locals() and img_rgb is not None:
                result = cv2.resize(img_rgb, (128, 128))
                face_cache[image_path] = result
                return result
        except:
            pass
        
        return None

# Define transformations for data augmentation
train_transforms = Compose([
    HorizontalFlip(p=0.5),

    RandomBrightnessContrast(p=0.2),
    ShiftScaleRotate(p=0.2),
    ColorJitter(p=0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transforms = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Custom dataset class
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        label = self.labels[idx]
        
        # Create the full path to the image
        if label == 0:  # Real image
            full_path = os.path.join(real_path, img_name)
        else:  # AI-generated image
            full_path = os.path.join(ai_path, img_name)
            
        # Check if file exists before processing
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        # Align face with the full path
        image = align_face(full_path)
        
        # If image is None after align_face, something went wrong with processing
        if image is None:
            raise RuntimeError(f"Failed to process image: {full_path}")

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

# Define CNN architecture
class FaceClassifierCNN(nn.Module):
    def __init__(self):
        super(FaceClassifierCNN, self).__init__()

        # Convolutional layers - solo 2 capas convolucionales
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

            # # Third convolutional block
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            # # Fourth convolutional block
            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            # # Fifth convolutional block
            # nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
        )

        # Calcular el tamaño de la salida antes de las capas fully connected
        # Para imágenes de 128x128, después de 2 capas de MaxPool2d (128/2/2 = 32)
        # Con 64 filtros, será 64 * 32 * 32 = 65536
        fc_input_size = 64 * 32 * 32
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_input_size, 512),  # Entrada desde las convoluciones
            nn.ReLU(),
            nn.Linear(512, 2)  # 2 clases: real o generado por IA
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# Function to prepare dataset
def prepare_dataset():
    # Get all image paths and filter for valid image types
    real_images = [img for img in os.listdir(real_path) 
                  if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    ai_images = [img for img in os.listdir(ai_path) 
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    print(f"Found {len(real_images)} real images and {len(ai_images)} AI-generated images")

    # Validate images before training
    print("Validating images...")
    valid_real_images = []
    valid_ai_images = []
    
    # Check real images
    for img in tqdm(real_images, desc="Validating real images"):
        full_path = os.path.join(real_path, img)
        if not os.path.exists(full_path):
            continue
            
        # Try to open the image to ensure it's valid
        test_img = cv2.imread(full_path)
        if test_img is None:
            print(f"Cannot read image: {full_path}")
            continue
            
        # Try processing the face
        face = align_face(full_path)
        if face is not None:
            valid_real_images.append(img)
    
    # Check AI images
    for img in tqdm(ai_images, desc="Validating AI images"):
        full_path = os.path.join(ai_path, img)
        if not os.path.exists(full_path):
            continue
            
        # Try to open the image to ensure it's valid
        test_img = cv2.imread(full_path)
        if test_img is None:
            print(f"Cannot read image: {full_path}")
            continue
            
        # Try processing the face
        face = align_face(full_path)
        if face is not None:
            valid_ai_images.append(img)
    
    print(f"After validation: {len(valid_real_images)} valid real images and {len(valid_ai_images)} valid AI images")
    
    # Only continue if we have enough valid images
    if len(valid_real_images) < 10 or len(valid_ai_images) < 10:
        raise ValueError("Not enough valid images to train the model!")

    # Create labels (0 for real, 1 for AI)
    real_labels = [0] * len(valid_real_images)
    ai_labels = [1] * len(valid_ai_images)

    # Combine datasets
    all_images = valid_real_images + valid_ai_images
    all_labels = real_labels + ai_labels

    # Split into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)  # 0.25 x 0.8 = 0.2

    # Create datasets
    train_dataset = FaceDataset(X_train, y_train, transform=train_transforms)
    val_dataset = FaceDataset(X_val, y_val, transform=val_transforms)
    test_dataset = FaceDataset(X_test, y_test, transform=val_transforms)

    # Use a smaller batch size to reduce memory usage
    batch_size = 32
    
    # Create dataloaders with no multiprocessing workers (use 0 to avoid issues)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        # Save statistics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'ours.pth')
            print(f'Model saved with validation accuracy: {val_acc:.4f}')

    return model, history

# Function to test the model
def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Additional metrics can be calculated using all_preds and all_labels
    return accuracy, all_preds, all_labels

# Plot training history
def plot_history(history, results_dir):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    plt.show()

def evaluate_on_new_dataset():
    """
    Evaluate the existing trained model on a different dataset (the one used by the finetuned models).
    
    This function:
    1. Loads the pre-trained 'ours.pth' model
    2. Evaluates it on the dataset from 'datasets/image/AI-Face-Detection'
    3. Generates metrics, confusion matrix, and other visualizations
    4. Saves results with '_v2' suffix to avoid overwriting existing files
    """
    # Use the existing results directory
    results_dir = "ourmodel_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    
    # Define paths to the new dataset
    DATASET_PATH = "datasets/image/AI-Face-Detection"
    REAL_PATH = os.path.join(DATASET_PATH, "real")
    FAKE_PATH = os.path.join(DATASET_PATH, "fake")
    
    # Check if dataset exists
    if not os.path.exists(REAL_PATH) or not os.path.exists(FAKE_PATH):
        print(f"Error: Dataset directories {REAL_PATH} or {FAKE_PATH} do not exist.")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the pre-trained model
    print("Loading pre-trained model...")
    model = FaceClassifierCNN()
    try:
        model.load_state_dict(torch.load('ourmodel_results/ours.pth', map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Prepare datasets
    print("Preparing datasets from new data...")
    real_images = [img for img in os.listdir(REAL_PATH) 
                  if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    fake_images = [img for img in os.listdir(FAKE_PATH) 
                  if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
    
    # Limit dataset size if too large (optional)
    MAX_SAMPLES = 1000  # adjust as needed
    if len(real_images) > MAX_SAMPLES:
        real_images = random.sample(real_images, MAX_SAMPLES)
    if len(fake_images) > MAX_SAMPLES:
        fake_images = random.sample(fake_images, MAX_SAMPLES)
    
    print(f"Using {len(real_images)} real images and {len(fake_images)} fake images")
    
    # Process and evaluate all images
    all_preds = []
    all_labels = []
    all_probs = []  # for probability distribution plotting
    
    print("Processing real images...")
    for img_name in tqdm(real_images):
        img_path = os.path.join(REAL_PATH, img_name)
        
        try:
            # Align face
            aligned_face = align_face(img_path)
            if aligned_face is None:
                print(f"Failed to process image: {img_path}")
                continue
                
            # Apply transformation
            augmented = val_transforms(image=aligned_face)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            # Store results (0 = real, 1 = fake)
            all_preds.append(predicted.item())
            all_labels.append(0)  # real label
            all_probs.append(probabilities[0, 1].item())  # probability of being fake
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print("Processing fake/AI images...")
    for img_name in tqdm(fake_images):
        img_path = os.path.join(FAKE_PATH, img_name)
        
        try:
            # Align face
            aligned_face = align_face(img_path)
            if aligned_face is None:
                print(f"Failed to process image: {img_path}")
                continue
                
            # Apply transformation
            augmented = val_transforms(image=aligned_face)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            # Store results (0 = real, 1 = fake)
            all_preds.append(predicted.item())
            all_labels.append(1)  # fake label
            all_probs.append(probabilities[0, 1].item())  # probability of being fake
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print metrics
    print("\n=== Model Performance on New Dataset ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate class-specific accuracy
    real_correct = cm[0, 0]
    real_total = cm[0, 0] + cm[0, 1]
    real_acc = real_correct / real_total if real_total > 0 else 0
    
    fake_correct = cm[1, 1]
    fake_total = cm[1, 0] + cm[1, 1]
    fake_acc = fake_correct / fake_total if fake_total > 0 else 0
    
    print(f"Real images accuracy: {real_acc:.4f}")
    print(f"AI-generated images accuracy: {fake_acc:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Real    AI-gen")
    print(f"Actual Real     {cm[0][0]:<7} {cm[0][1]}")
    print(f"       AI-gen   {cm[1][0]:<7} {cm[1][1]}")
    
    # Save metrics to text file
    metrics_path = os.path.join(results_dir, 'model_metrics_v2.txt')
    with open(metrics_path, 'w') as f:
        f.write("=== Our CNN Model Performance on New Dataset ===\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        
        f.write(f"Real images accuracy: {real_acc:.4f}\n")
        f.write(f"AI-generated images accuracy: {fake_acc:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("                 Predicted\n")
        f.write("                 Real    AI-gen\n")
        f.write(f"Actual Real     {cm[0][0]:<7} {cm[0][1]}\n")
        f.write(f"       AI-gen   {cm[1][0]:<7} {cm[1][1]}\n\n")
        
        # Additional Statistics
        f.write("Additional Statistics:\n")
        f.write(f"Total images: {len(all_preds)}\n")
        f.write(f"Correct predictions: {sum(np.array(all_preds) == np.array(all_labels))}\n")
        f.write(f"Incorrect predictions: {sum(np.array(all_preds) != np.array(all_labels))}\n\n")
        
        # Count predictions by class
        real_pred_count = sum(1 for p in all_preds if p == 0)
        fake_pred_count = sum(1 for p in all_preds if p == 1)
        
        f.write(f"Images predicted as real: {real_pred_count} ({real_pred_count/len(all_preds)*100:.2f}%)\n")
        f.write(f"Images predicted as AI-generated: {fake_pred_count} ({fake_pred_count/len(all_preds)*100:.2f}%)\n\n")
        
        # Dataset information
        f.write("=== Dataset Information ===\n\n")
        real_count = sum(1 for l in all_labels if l == 0)
        fake_count = sum(1 for l in all_labels if l == 1)
        f.write(f"Real images in dataset: {real_count}\n")
        f.write(f"AI-generated images in dataset: {fake_count}\n\n")
        
        # Model information
        f.write("=== Model Information ===\n\n")
        f.write("Architecture: 2-layer CNN with fully connected layers\n")
        total_params = sum(p.numel() for p in model.parameters())
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Images with no landmarks detected: {no_landmarks}\n")
    
    print(f"Metrics saved to: {metrics_path}")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'AI-gen'], 
                yticklabels=['Real', 'AI-gen'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - New Dataset')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_v2.png'))
    plt.close()
    
    # Create distribution plot of fake probabilities
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by true label
    real_probs = [prob for prob, label in zip(all_probs, all_labels) if label == 0]
    fake_probs = [prob for prob, label in zip(all_probs, all_labels) if label == 1]
    
    plt.hist(real_probs, bins=20, alpha=0.5, label='Real Images')
    plt.hist(fake_probs, bins=20, alpha=0.5, label='AI-Generated Images')
    plt.xlabel('Probability of Being AI-Generated')
    plt.ylabel('Count')
    plt.title('Distribution of Predictions on New Dataset')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'predictions_distribution_v2.png'))
    plt.close()
    
    print(f"Visualizations saved to: {results_dir}")
    return accuracy

# Main function to run the training and evaluation
def main():
    # # Create results directory
    # results_dir = "ourmodel_results"
    # os.makedirs(results_dir, exist_ok=True)
    # print(f"Results will be saved in: {results_dir}")
    
    # print("Preparing datasets...")
    # try:
    #     train_loader, val_loader, test_loader = prepare_dataset()
    # except Exception as e:
    #     print(f"Error preparing dataset: {str(e)}")
    #     return

    # print("Creating model...")
    # model = FaceClassifierCNN()

    # # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # print("Starting training...")
    # model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # # Save model
    # model_path = os.path.join(results_dir, 'ours.pth')
    # torch.save(model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")

    # # Plot training history
    # plot_history(history, results_dir)

    # # Load best model for testing
    # best_model = FaceClassifierCNN()
    # best_model.load_state_dict(torch.load(os.path.join(results_dir, 'ours.pth')))

    # print("Testing model...")
    # accuracy, preds, labels = test_model(best_model, test_loader)
    
    # # Calculate additional metrics
    # precision = precision_score(labels, preds, average='binary')
    # recall = recall_score(labels, preds, average='binary')
    # f1 = f1_score(labels, preds, average='binary')
    # cm = confusion_matrix(labels, preds)
    
    # # Save metrics to text file
    # metrics_path = os.path.join(results_dir, 'model_metrics.txt')
    # with open(metrics_path, 'w') as f:
    #     f.write("=== Our CNN Model Performance ===\n\n")
    #     f.write(f"Accuracy: {accuracy:.4f}\n")
    #     f.write(f"Precision: {precision:.4f}\n")
    #     f.write(f"Recall: {recall:.4f}\n")
    #     f.write(f"F1 Score: {f1:.4f}\n\n")
        
    #     # Calculate per-class accuracy
    #     real_correct = cm[0, 0]
    #     real_total = cm[0, 0] + cm[0, 1]
    #     real_acc = real_correct / real_total if real_total > 0 else 0
        
    #     fake_correct = cm[1, 1]
    #     fake_total = cm[1, 0] + cm[1, 1]
    #     fake_acc = fake_correct / fake_total if fake_total > 0 else 0
        
    #     f.write(f"Real images accuracy: {real_acc:.4f}\n")
    #     f.write(f"AI-generated images accuracy: {fake_acc:.4f}\n\n")
        
    #     f.write("Confusion Matrix:\n")
    #     f.write("                 Predicted\n")
    #     f.write("                 Real    AI-gen\n")
    #     f.write(f"Actual Real     {cm[0][0]:<7} {cm[0][1]}\n")
    #     f.write(f"       AI-gen   {cm[1][0]:<7} {cm[1][1]}\n\n")
        
    #     f.write("Additional Statistics:\n")
    #     f.write(f"Total images: {len(labels)}\n")
    #     f.write(f"Correct predictions: {sum(np.array(preds) == np.array(labels))}\n")
    #     f.write(f"Incorrect predictions: {sum(np.array(preds) != np.array(labels))}\n\n")
        
    #     # Count images predicted as each class
    #     real_pred_count = sum(1 for p in preds if p == 0)
    #     fake_pred_count = sum(1 for p in preds if p == 1)
        
    #     f.write(f"Images predicted as real: {real_pred_count} ({real_pred_count/len(labels)*100:.2f}%)\n")
    #     f.write(f"Images predicted as AI-generated: {fake_pred_count} ({fake_pred_count/len(labels)*100:.2f}%)\n\n")
        
    #     # Dataset information
    #     f.write("=== Dataset Information ===\n\n")
    #     real_count = sum(1 for l in labels if l == 0)
    #     fake_count = sum(1 for l in labels if l == 1)
    #     f.write(f"Real images in test set: {real_count}\n")
    #     f.write(f"AI-generated images in test set: {fake_count}\n\n")
        
    #     # Model information
    #     f.write("=== Model Information ===\n\n")
    #     f.write("Architecture: 2-layer CNN with fully connected layers\n")
    #     total_params = sum(p.numel() for p in model.parameters())
    #     f.write(f"Total parameters: {total_params:,}\n")
    #     f.write(f"Images with no landmarks detected: {no_landmarks}\n")
    
    # print(f"Metrics saved to: {metrics_path}")
    
    # # Create confusion matrix visualization
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    #             xticklabels=['Real', 'AI-gen'], 
    #             yticklabels=['Real', 'AI-gen'])
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    # plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    # plt.close()
    
    # print(f"Final test accuracy: {accuracy:.4f}")
    # print(f"Statistics:")
    # print(f"  - Images with no landmarks detected: {no_landmarks}")
    # print(f"All results saved to {results_dir}")
    
    # return accuracy, model
    evaluate_on_new_dataset()

if __name__ == "__main__":
    main()
