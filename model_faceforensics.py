import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import random
import kagglehub
import shutil
from tqdm import tqdm
import warnings
from torch.utils.model_zoo import load_url
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the models directory to the path so imports work
sys.path.append(os.path.abspath("."))

# Import required FaceForensics modules
from models.faceforensics_image.blazeface import FaceExtractor, BlazeFace
from models.faceforensics_image.architectures import fornet, weights
from models.faceforensics_image.isplutils import utils

# Define paths to the dataset and limit images for testing
DATASET_PATH = "datasets/image/AI-Face-Detection"
REAL_PATH = os.path.join(DATASET_PATH, "real")
FAKE_PATH = os.path.join(DATASET_PATH, "fake")
MAX_IMAGES_PER_CLASS = 1000

# Cache for processed faces
face_cache = {}

# Global variables for model to avoid reloading
faceforensics_model = None
face_extractor = None
transformer = None
device = None

def load_faceforensics_model():
    """
    Load FaceForensics model once and store in global variables
    """
    global faceforensics_model, face_extractor, transformer, device
    
    # Choose model architecture and training dataset
    net_model = 'EfficientNetAutoAttB4'
    train_db = 'DFDC'
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    face_policy = 'scale'
    face_size = 224
    
    print("Loading FaceForensics model...")
    
    # Set up model
    model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
    faceforensics_model = getattr(fornet, net_model)().eval().to(device)
    faceforensics_model.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
    print("Loaded pretrained weights for", net_model)
    
    # Set up transformer
    transformer = utils.get_transformer(face_policy, face_size, faceforensics_model.get_normalizer(), train=False)
    
    # Face detection
    facedet = BlazeFace().to(device)
    facedet.load_weights("models/faceforensics_image/blazeface/blazeface.pth")
    facedet.load_anchors("models/faceforensics_image/blazeface/anchors.npy")
    face_extractor = FaceExtractor(facedet=facedet)
    print("Face detector loaded")

def predictor_Faceforensics(img_input, save_result=True):
    """
    FaceForensics prediction function integrated into the main file.
    Can accept either a path to an image or a numpy array containing an image.
    
    Parameters:
    - img_input: Either a path to an image file or a numpy array with RGB image data
    - save_result: Whether to save the result to a file (default: True)
    
    Returns:
    - Probability that the image is REAL (0.0-1.0)
    """
    global faceforensics_model, face_extractor, transformer, device
    
    # Make sure model is loaded
    if faceforensics_model is None:
        load_faceforensics_model()
    
    # Load the image based on input type
    if isinstance(img_input, str):
        # Input is a file path
        im_original = Image.open(img_input)
        if im_original.mode != 'RGB':
            im_original = im_original.convert('RGB')
    else:
        # Input is a numpy array
        im_original = Image.fromarray(img_input.astype('uint8'))
        if im_original.mode != 'RGB':
            im_original = im_original.convert('RGB')

    # Process the image to extract faces
    try:
        im_original_face = face_extractor.process_image(img=im_original)
        
        # Check if any faces were detected
        if 'faces' not in im_original_face or len(im_original_face['faces']) == 0:
            if isinstance(img_input, str):
                print(f"Warning: No face detected in {img_input}")
            else:
                print("Warning: No face detected in provided image array")
            return 0.5  # Neutral value when no face is detected
            
        im_original_face = im_original_face['faces'][0]  # take the face with the highest confidence score
        
        # Ensure the extracted face is in RGB format
        if im_original_face.shape[2] != 3:
            # Convert 2-channel to 3-channel by duplicating channels
            if im_original_face.shape[2] == 2:
                # Create a 3-channel image by duplicating one of the channels
                rgb_face = np.zeros((im_original_face.shape[0], im_original_face.shape[1], 3), dtype=np.uint8)
                rgb_face[:, :, 0] = im_original_face[:, :, 0]  # R channel
                rgb_face[:, :, 1] = im_original_face[:, :, 0]  # G channel (duplicate)
                rgb_face[:, :, 2] = im_original_face[:, :, 1]  # B channel
                im_original_face = rgb_face
            # Convert 1-channel grayscale to 3-channel RGB
            elif im_original_face.shape[2] == 1:
                im_original_face = np.repeat(im_original_face, 3, axis=2)
            
    except Exception as e:
        if isinstance(img_input, str):
            print(f"Error processing {img_input}: {str(e)}")
        else:
            print(f"Error processing image array: {str(e)}")
        return 0.5  # Neutral value when processing fails

    # Prepare the face for the model
    try:
        faces_t = torch.stack([transformer(image=im)['image'] for im in [im_original_face, im_original_face]])
    except Exception as e:
        if isinstance(img_input, str):
            print(f"Error transforming face from {img_input}: {str(e)}")
        else:
            print(f"Error transforming face from image array: {str(e)}")
        return 0.5  # Neutral value when transformation fails

    # Get prediction
    with torch.no_grad():
        faces_pred = torch.sigmoid(faceforensics_model(faces_t.to(device))).cpu().numpy().flatten()

    """
    Interpret scores.
    A score close to 0 predicts REAL. A score close to 1 predicts FAKE.
    But we invert it to return probability that the image is REAL.
    """
    probab = (1 - float('{:.4f}'.format(faces_pred[0])))
    
    # Save result to a file if requested
    if save_result:
        results_dir = "faceforensics_results"
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "result.txt"), "w") as text_file:
            text_file.write(str(probab))
        
    return probab

def download_dataset():
    """Download and setup the dataset if it doesn't exist."""
    if os.path.exists(DATASET_PATH):
        print(f"Dataset already exists at {DATASET_PATH}")
        return
    
    print("Downloading dataset...")
    download_path = kagglehub.dataset_download("shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset")
    real_path = os.path.join(download_path, "AI-face-detection-Dataset", "real")
    ai_path = os.path.join(download_path, "AI-face-detection-Dataset", "AI")
    
    # Create directories
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("datasets/image", exist_ok=True)
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(REAL_PATH, exist_ok=True)
    os.makedirs(FAKE_PATH, exist_ok=True)
    
    # Copy real images
    print("Copying real images...")
    for img_file in os.listdir(real_path):
        src_file = os.path.join(real_path, img_file)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(REAL_PATH, img_file))
    
    # Copy AI/fake images
    print("Copying fake/AI images...")
    for img_file in os.listdir(ai_path):
        src_file = os.path.join(ai_path, img_file)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(FAKE_PATH, img_file))
    
    print(f"Dataset setup complete. {len(os.listdir(REAL_PATH))} real images, {len(os.listdir(FAKE_PATH))} fake images.")

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
            result = cv2.resize(img_rgb, (224, 224))
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
        face_img = cv2.resize(face_img, (224, 224))
        
        # Save to cache
        face_cache[image_path] = face_img
        return face_img
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        try:
            # If error occurs, try to return resized original
            result = cv2.resize(img_rgb, (224, 224))
            face_cache[image_path] = result
            return result
        except:
            return None

def process_images(directory, label, max_images=MAX_IMAGES_PER_CLASS):
    """Process images in a directory and return predictions with actual labels."""
    results = []
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return results
    
    image_files = [f for f in os.listdir(directory) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {directory}")
        return results
    
    # Limit the number of images for quick testing
    if max_images > 0 and len(image_files) > max_images:
        image_files = random.sample(image_files, max_images)
    
    print(f"Processing {len(image_files)} images from {directory}...")
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(directory, img_file)
        
        try:
            # Apply face detection and alignment
            aligned_face = align_face(img_path)
            if aligned_face is None:
                print(f"Failed to process image: {img_path}")
                continue
            
            # Now directly use the aligned face array with the predictor
            # No need to save to a temporary file
            probability = predictor_Faceforensics(aligned_face, save_result=False)
            
            # Determine prediction (>0.5 = real, <0.5 = fake)
            predicted_label = "real" if probability > 0.5 else "fake"
            
            # Calculate confidence
            confidence = probability if predicted_label == "real" else (1 - probability)
            
            results.append({
                'image_path': img_path,
                'actual_label': label,
                'predicted_label': predicted_label,
                'probability': probability,
                'confidence': confidence,
                'correct': (label == predicted_label)
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    return results

def calculate_metrics(results_df, results_dir):
    """Calculate and display metrics for the model performance."""
    y_true = [1 if label == "real" else 0 for label in results_df['actual_label']]
    y_pred = [1 if label == "real" else 0 for label in results_df['predicted_label']]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display metrics
    print("\n=== Model Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate class-specific accuracy
    real_acc = results_df[results_df['actual_label'] == 'real']['correct'].mean()
    fake_acc = results_df[results_df['actual_label'] == 'fake']['correct'].mean()
    
    print(f"Real images accuracy: {real_acc:.4f}")
    print(f"Fake images accuracy: {fake_acc:.4f}")
    
    # Display confusion matrix in text format
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("               Predicted")
    print("               Fake    Real")
    print(f"Actual Fake    {cm[0][0]:<7} {cm[0][1]}")
    print(f"       Real    {cm[1][0]:<7} {cm[1][1]}")
    
    # Calculate additional statistics
    print("\nAdditional Statistics:")
    print(f"Total images: {len(results_df)}")
    print(f"Correct predictions: {sum(results_df['correct'])}")
    print(f"Incorrect predictions: {len(results_df) - sum(results_df['correct'])}")
    
    real_pred_count = sum(results_df['predicted_label'] == 'real')
    fake_pred_count = sum(results_df['predicted_label'] == 'fake')
    print(f"Images predicted as real: {real_pred_count} ({real_pred_count/len(results_df)*100:.2f}%)")
    print(f"Images predicted as fake: {fake_pred_count} ({fake_pred_count/len(results_df)*100:.2f}%)")
    
    # Output example predictions
    print("\nSample predictions (first 5):")
    for i, row in results_df.head(5).iterrows():
        img_name = os.path.basename(row['image_path'])
        print(f"{img_name}: Actual: {row['actual_label']}, Predicted: {row['predicted_label']}, Confidence: {row['confidence']:.4f}")
    
    # Save metrics to a text file
    metrics_file = os.path.join(results_dir, "model_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("=== FaceForensics Model Performance ===\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        
        f.write(f"Real images accuracy: {real_acc:.4f}\n")
        f.write(f"Fake images accuracy: {fake_acc:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("               Predicted\n")
        f.write("               Fake    Real\n")
        f.write(f"Actual Fake    {cm[0][0]:<7} {cm[0][1]}\n")
        f.write(f"       Real    {cm[1][0]:<7} {cm[1][1]}\n\n")
        
        f.write("Additional Statistics:\n")
        f.write(f"Total images: {len(results_df)}\n")
        f.write(f"Correct predictions: {sum(results_df['correct'])}\n")
        f.write(f"Incorrect predictions: {len(results_df) - sum(results_df['correct'])}\n\n")
        
        f.write(f"Images predicted as real: {real_pred_count} ({real_pred_count/len(results_df)*100:.2f}%)\n")
        f.write(f"Images predicted as fake: {fake_pred_count} ({fake_pred_count/len(results_df)*100:.2f}%)\n\n")
        
        # Dataset information
        f.write("=== Dataset Information ===\n\n")
        f.write(f"MAX_IMAGES_PER_CLASS: {MAX_IMAGES_PER_CLASS}\n")
        real_count = len([x for x in results_df['actual_label'] if x == 'real'])
        fake_count = len([x for x in results_df['actual_label'] if x == 'fake'])
        f.write(f"Real images: {real_count}\n")
        f.write(f"Fake images: {fake_count}\n")
    
    print(f"Metrics saved to: {metrics_file}")
    
    # Create and save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    # Save plot
    cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_plot_path}")
    
    # Create and save distribution plot
    confidence_real = results_df[results_df['actual_label'] == 'real']['probability']
    confidence_fake = results_df[results_df['actual_label'] == 'fake']['probability']
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_real, bins=20, alpha=0.5, label='Real Images')
    plt.hist(confidence_fake, bins=20, alpha=0.5, label='Fake Images')
    plt.xlabel('Probability of Being Real')
    plt.ylabel('Count')
    plt.title('Distribution of Predictions')
    plt.legend()
    
    # Save plot
    dist_plot_path = os.path.join(results_dir, "predictions_distribution.png")
    plt.savefig(dist_plot_path)
    plt.close()
    print(f"Predictions distribution plot saved to: {dist_plot_path}")

def main():
    # Create results directory
    results_dir = "faceforensics_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    
    # Download and setup dataset if it doesn't exist
    download_dataset()
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset directory {DATASET_PATH} does not exist.")
        return
    
    # Load model once before processing any images
    load_faceforensics_model()
    
    # Process real images
    real_results = process_images(REAL_PATH, "real")
    
    # Process fake images
    fake_results = process_images(FAKE_PATH, "fake")
    
    # Combine results
    all_results = real_results + fake_results
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("No valid results to analyze. Please check your dataset and preprocessing.")
        return
    
    # Calculate and display metrics
    calculate_metrics(results_df, results_dir)

if __name__ == "__main__":
    main() 



    