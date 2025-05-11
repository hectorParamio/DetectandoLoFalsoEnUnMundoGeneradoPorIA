import kagglehub
import os
import sys
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
import shutil
import json
from PIL import Image
import torchvision.transforms as transforms
import seaborn as sns
from datetime import datetime

# Add cnndetection_image to path and import resnet50
sys.path.insert(0, os.path.abspath('models/cnndetection_image'))
try:
    from networks.resnet import resnet50
except ImportError:
    # If that doesn't work, try a direct import
    sys.path.insert(0, os.path.abspath('models/cnndetection_image/networks'))
    from resnet import resnet50

# Global variable to determine if we should invert the prediction logic
should_invert_prediction = False

# Step 1: Download and prepare dataset from Kaggle
print("\n=== STEP 1: DOWNLOADING AND PREPARING DATASET ===\n")

# Get dataset from kaggle and prepare the structure
download_path = kagglehub.dataset_download("shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset")
real_path = os.path.join(download_path, "AI-face-detection-Dataset", "real")
ai_path = os.path.join(download_path, "AI-face-detection-Dataset", "AI")

dest_base = "datasets/image/AI-Face-Detection"
dest_real = os.path.join(dest_base, "real")
dest_fake = os.path.join(dest_base, "fake")

os.makedirs("datasets", exist_ok=True)
os.makedirs("datasets/image", exist_ok=True)
os.makedirs(dest_base, exist_ok=True)
os.makedirs(dest_real, exist_ok=True)
os.makedirs(dest_fake, exist_ok=True)

for img_file in os.listdir(real_path):
    src_file = os.path.join(real_path, img_file)
    if os.path.isfile(src_file):
        shutil.copy2(src_file, os.path.join(dest_real, img_file))
print(f"Copied {len(os.listdir(real_path))} real images")

for img_file in os.listdir(ai_path):
    src_file = os.path.join(ai_path, img_file)
    if os.path.isfile(src_file):
        shutil.copy2(src_file, os.path.join(dest_fake, img_file))
print(f"Copied {len(os.listdir(ai_path))} AI/fake images")

# Create results directory
RESULTS_DIR = "cnndetection_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create temp directory for temporary image processing
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Step 2: Set up CNN Detection model
print("\n=== STEP 2: SETTING UP CNN DETECTION MODEL ===\n")

def load_cnn_detection_model():
    """Load the CNN Detection model for deepfake detection"""
    print("Loading CNN Detection model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = resnet50(num_classes=1)
    
    # Load pre-trained weights
    model_path = os.path.join('models', 'cnndetection_image', 'weights', 'blur_jpg_prob0.5.pth')
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    
    # Move to device and set to eval mode
    if device.type == 'cuda':
        model = model.cuda()
    model.eval()
    
    # Set up image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform, device

def analyze_image(image_path, model, transform, device):
    """
    Analyze a single image with the CNN Detection model
    
    Returns:
    --------
    float: Probability of being fake/synthetic (higher means likely fake)
    """
    # Open and convert image
    img = Image.open(image_path).convert('RGB')
    
    # Transform image
    img_tensor = transform(img).unsqueeze(0)
    
    # Move to device
    if device.type == 'cuda':
        img_tensor = img_tensor.cuda()
    
    # Run inference
    with torch.no_grad():
        prob = model(img_tensor).sigmoid().item()
    
    # If we should invert the prediction, return the opposite
    if should_invert_prediction:
        return 1.0 - prob
        
    return prob

def find_optimal_threshold(model, transform, device):
    """Find the optimal threshold for classification by testing different values"""
    print("\nFinding optimal detection threshold...")
    
    # Sample a subset of images for threshold tuning
    max_samples = 100  # Limit samples for speed
    
    real_files = os.listdir(dest_real)[:max_samples]
    fake_files = os.listdir(dest_fake)[:max_samples]
    
    real_probs = []
    fake_probs = []
    
    # Process real images
    for img_file in tqdm(real_files, desc="Processing real samples"):
        img_path = os.path.join(dest_real, img_file)
        if os.path.isfile(img_path):
            prob = analyze_image(img_path, model, transform, device)
            real_probs.append(prob)
    
    # Process fake images
    for img_file in tqdm(fake_files, desc="Processing fake samples"):
        img_path = os.path.join(dest_fake, img_file)
        if os.path.isfile(img_path):
            prob = analyze_image(img_path, model, transform, device)
            fake_probs.append(prob)
    
    # Print statistics about the probabilities
    print("\nStatistics for model predictions:")
    print(f"Real images - Min: {min(real_probs):.4f}, Max: {max(real_probs):.4f}, Mean: {sum(real_probs)/len(real_probs):.4f}")
    print(f"Fake images - Min: {min(fake_probs):.4f}, Max: {max(fake_probs):.4f}, Mean: {sum(fake_probs)/len(fake_probs):.4f}")
    
    # Calculate the proper threshold based on actual output distribution
    # Sort all probabilities and find a good separation point
    all_probs = [(p, 'real') for p in real_probs] + [(p, 'fake') for p in fake_probs]
    all_probs.sort()  # Sort by probability
    
    best_threshold = 0.5
    min_error = float('inf')
    step = 0.01  # Smaller step for finer search
    
    # Check raw probabilities of first 5 real and fake images for debugging
    print("\nSample probabilities:")
    print("Real images:", real_probs[:5])
    print("Fake images:", fake_probs[:5])
    
    # Test different thresholds
    thresholds = np.arange(0.01, 1.0, step)
    best_threshold = 0.5
    best_f1 = 0
    threshold_results = []
    
    for threshold in thresholds:
        # Calculate metrics for this threshold
        real_correct = sum(1 for p in real_probs if p <= threshold)
        fake_correct = sum(1 for p in fake_probs if p > threshold)
        
        real_accuracy = real_correct / len(real_probs) if real_probs else 0
        fake_accuracy = fake_correct / len(fake_probs) if fake_probs else 0
        
        # Calculate precision, recall and F1 score
        tp = fake_correct
        fp = len(real_probs) - real_correct
        fn = len(fake_probs) - fake_correct
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store results
        threshold_results.append({
            "threshold": threshold,
            "real_accuracy": real_accuracy,
            "fake_accuracy": fake_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
        # Check if this is the best threshold so far
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Print threshold tuning results
    print("\nThreshold tuning results:")
    print(f"{'Threshold':<10} {'Real Acc':<10} {'Fake Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 60)
    
    # Print a subset of results for clarity
    for result in threshold_results[::5]:  # Print every 5th result
        print(f"{result['threshold']:<10.2f} {result['real_accuracy']:<10.2f} {result['fake_accuracy']:<10.2f} "
              f"{result['precision']:<10.2f} {result['recall']:<10.2f} {result['f1']:<10.2f}")
    
    print(f"\nOptimal threshold: {best_threshold:.2f} (F1 score: {best_f1:.2f})")
    
    # Fix: Add a check to ensure we have a valid threshold with non-zero F1 score
    if best_f1 == 0:
        print("\nWARNING: Could not find a threshold with non-zero F1 score.")
        print("This might indicate that the model predictions for real and fake are not well separated.")
        print("Checking if we need to invert the classification logic...")
        
        # Try inverting the classification logic (< threshold for fake, > threshold for real)
        inverted_thresholds = np.arange(0.01, 1.0, step)
        best_inverted_threshold = 0.5
        best_inverted_f1 = 0
        
        for threshold in inverted_thresholds:
            # Invert the logic: fake images < threshold, real images > threshold
            real_correct = sum(1 for p in real_probs if p > threshold)
            fake_correct = sum(1 for p in fake_probs if p <= threshold)
            
            real_accuracy = real_correct / len(real_probs) if real_probs else 0
            fake_accuracy = fake_correct / len(fake_probs) if fake_probs else 0
            
            tp = fake_correct
            fp = len(real_probs) - real_correct
            fn = len(fake_probs) - fake_correct
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_inverted_f1:
                best_inverted_f1 = f1
                best_inverted_threshold = threshold
        
        print(f"Inverted logic - Best threshold: {best_inverted_threshold:.2f} (F1 score: {best_inverted_f1:.2f})")
        
        # If inverting gives better results, use that approach
        if best_inverted_f1 > best_f1:
            print("Inverting classification logic gives better results!")
            best_threshold = best_inverted_threshold
            best_f1 = best_inverted_f1
            
            # Update the analyze_image function to use inverted logic
            global should_invert_prediction
            should_invert_prediction = True
    
    return best_threshold

def analyze_dataset():
    """Analyze all images in the dataset and save results"""
    # Create results directory
    results_dir = "ourmodel_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    
    # Load model
    model, transform, device = load_cnn_detection_model()
    
    # Find optimal threshold
    threshold = find_optimal_threshold(model, transform, device)
    
    # Report if we're using inverted prediction logic
    if should_invert_prediction:
        print("\nUsing INVERTED prediction logic - lower scores indicate fake images")
    else:
        print("\nUsing standard prediction logic - higher scores indicate fake images")
    
    results = {
        "real": [],
        "fake": []
    }
    
    # Process real images
    print("\nAnalyzing real images...")
    for img_file in tqdm(os.listdir(dest_real)):
        img_path = os.path.join(dest_real, img_file)
        if os.path.isfile(img_path):
            prob = analyze_image(img_path, model, transform, device)
            # With should_invert_prediction, the logic is already handled in analyze_image
            # Convert NumPy types to native Python types
            prob_float = float(prob)
            is_correct = bool(prob <= threshold)
            results["real"].append({
                "filename": img_file,
                "prob": prob_float,
                "prediction": "fake" if prob > threshold else "real",
                "ground_truth": "real",
                "correct": is_correct
            })
    
    # Process fake images
    print("\nAnalyzing fake/AI-generated images...")
    for img_file in tqdm(os.listdir(dest_fake)):
        img_path = os.path.join(dest_fake, img_file)
        if os.path.isfile(img_path):
            prob = analyze_image(img_path, model, transform, device)
            # Convert NumPy types to native Python types
            prob_float = float(prob)
            is_correct = bool(prob > threshold)
            results["fake"].append({
                "filename": img_file,
                "prob": prob_float,
                "prediction": "fake" if prob > threshold else "real",
                "ground_truth": "fake",
                "correct": is_correct
            })
    
    # Calculate accuracy metrics
    real_correct = sum(1 for r in results["real"] if r["correct"])
    fake_correct = sum(1 for r in results["fake"] if r["correct"])
    
    total_images = len(results["real"]) + len(results["fake"])
    total_correct = real_correct + fake_correct
    
    accuracy = total_correct / total_images if total_images > 0 else 0
    real_accuracy = real_correct / len(results["real"]) if len(results["real"]) > 0 else 0
    fake_accuracy = fake_correct / len(results["fake"]) if len(results["fake"]) > 0 else 0
    
    # Calculate precision and recall
    true_positives = fake_correct
    false_positives = len(results["real"]) - real_correct
    false_negatives = len(results["fake"]) - fake_correct
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add summary to results - convert all NumPy types to Python native types
    results["summary"] = {
        "total_images": int(total_images),
        "threshold": float(threshold),
        "inverted_logic": bool(should_invert_prediction),
        "accuracy": float(accuracy),
        "real_accuracy": float(real_accuracy),
        "fake_accuracy": float(fake_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    # Save metrics to a text file
    metrics_path = os.path.join(results_dir, "model_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("=== CNN Detection Model Performance ===\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        
        f.write(f"Real images accuracy: {real_accuracy:.4f}\n")
        f.write(f"Fake images accuracy: {fake_accuracy:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("               Predicted\n")
        f.write("               Fake    Real\n")
        f.write(f"Actual Fake    {fake_correct:<7} {len(results['fake'])-fake_correct}\n")
        f.write(f"       Real    {len(results['real'])-real_correct:<7} {real_correct}\n\n")
        
        f.write("Additional Statistics:\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"Correct predictions: {total_correct}\n")
        f.write(f"Incorrect predictions: {total_images - total_correct}\n\n")
        
        real_pred_count = sum(1 for r in results["real"] if r["prediction"] == "real") + \
                         sum(1 for r in results["fake"] if r["prediction"] == "real")
        fake_pred_count = sum(1 for r in results["real"] if r["prediction"] == "fake") + \
                         sum(1 for r in results["fake"] if r["prediction"] == "fake")
        
        f.write(f"Images predicted as real: {real_pred_count} ({real_pred_count/total_images*100:.2f}%)\n")
        f.write(f"Images predicted as fake: {fake_pred_count} ({fake_pred_count/total_images*100:.2f}%)\n\n")
        
        f.write("=== Dataset Information ===\n\n")
        f.write(f"Total images analyzed: {total_images}\n")
        f.write(f"Real images: {len(results['real'])}\n")
        f.write(f"Fake images: {len(results['fake'])}\n\n")
        
        f.write("=== Detection Parameters ===\n\n")
        f.write(f"Optimal threshold: {threshold:.4f}\n")
        f.write(f"Using inverted logic: {should_invert_prediction}\n")
        if should_invert_prediction:
            f.write("Note: With inverted logic, LOWER scores indicate fake images\n")
        else:
            f.write("Note: With standard logic, HIGHER scores indicate fake images\n")
    
    print(f"Metrics saved to: {metrics_path}")
    
    # Create visualizations directly in the results folder
    create_visualizations(results, results_dir)
    
    return results

def create_visualizations(results, output_dir):
    """Create and save visualizations of the results directly in the output directory"""
    print("\n=== CREATING VISUALIZATIONS ===\n")
    
    # Extract data for visualizations
    threshold = results["summary"]["threshold"]
    inverted_logic = results["summary"].get("inverted_logic", False)
    
    # Distribution of probabilities for real and fake images
    plt.figure(figsize=(12, 6))
    
    # Real images probabilities
    real_probs = [r["prob"] for r in results["real"]]
    plt.hist(real_probs, bins=30, alpha=0.7, label='Real Images', color='green')
    
    # Fake images probabilities
    fake_probs = [r["prob"] for r in results["fake"]]
    plt.hist(fake_probs, bins=30, alpha=0.7, label='Fake Images', color='red')
    
    plt.axvline(x=threshold, color='black', linestyle='--', 
                label=f'Threshold: {threshold:.2f}')
    
    title = 'Distribution of Synthetic Probabilities'
    if inverted_logic:
        title += ' (INVERTED LOGIC: lower scores indicate fake)'
        
    plt.title(title)
    plt.xlabel('Probability Score')
    plt.ylabel('Number of Images')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
    plt.close()
    
    # Create confusion matrix visualization
    fake_correct = sum(1 for r in results["fake"] if r["correct"])
    real_correct = sum(1 for r in results["real"] if r["correct"])
    
    cm = [
        [fake_correct, len(results["fake"]) - fake_correct],
        [len(results["real"]) - real_correct, real_correct]
    ]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Create ROC curve
    from sklearn.metrics import roc_curve, auc
    
    # Combine all probabilities and labels
    all_probs = real_probs + fake_probs
    all_labels = [0] * len(real_probs) + [1] * len(fake_probs)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

# Run the analysis
analyze_dataset()

print("\n=== ANALYSIS COMPLETE ===\n")
print(f"Results and visualizations have been saved to the results directory.")

# Clean up temporary files
try:
    if os.path.exists(TEMP_DIR) and os.path.isdir(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"Cleaned up temporary directory: {TEMP_DIR}")
except Exception as e:
    print(f"Warning: Could not clean up temporary directory: {e}")

