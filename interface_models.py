import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add models to the path
sys.path.append(os.path.abspath("."))

# Try importing model modules
try:
    # Import our custom model
    from models_ours_finetunned import FaceClassifierCNN, align_face as our_align_face
    
    # Import FaceForensics modules
    from models.faceforensics_image.blazeface import FaceExtractor, BlazeFace
    from models.faceforensics_image.architectures import fornet, weights
    from torch.utils.model_zoo import load_url
    
    # Import CNN Detection model
    sys.path.insert(0, os.path.abspath('models/cnndetection_image'))
    from networks.resnet import resnet50
    
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing model modules: {e}")
    MODELS_AVAILABLE = False

# Model paths - update these as needed
MODEL_PATHS = {
    "Our Model": "ourmodel_results/ours.pth",
    "Our Model (Fine-tuned)": "ourmodel_finetunned_results/ours_finetunned.pth",
    "CNN Detection": "models/cnndetection_image/weights/blur_jpg_prob0.5.pth",
    "FaceForensics": "models/faceforensics_image/weights/EfficientNetAutoAttB4_DFDC.pth",
    "FaceForensics (Fine-tuned)": "faceforensics_finetunned_results/faceforensics_finetuned.pth"
}

# Fixed image size constants
DISPLAY_IMAGE_WIDTH = 350
DISPLAY_IMAGE_HEIGHT = 350

# Model wrapper classes for consistent API
class OurModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def predict(self, img):
        """Predict whether an image is real or fake using our custom model"""
        # Save image temporarily to process with align_face
        temp_path = "temp/temp_image.jpg"
        img.save(temp_path)
        
        try:
            # Process with align_face
            processed_face = our_align_face(temp_path)
            if processed_face is None:
                # If face alignment fails, resize the original
                img_np = np.array(img)
                processed_face = cv2.resize(img_np, (128, 128))
        except Exception as e:
            print(f"Error in face alignment: {e}")
            # Fallback to simple resize
            img_np = np.array(img)
            processed_face = cv2.resize(img_np, (128, 128))
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert array to PIL for transform
        pil_image = Image.fromarray(processed_face)
        tensor = transform(pil_image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            fake_prob = probs[0][1].item()  # Probability of being fake
        
        return fake_prob

class CNNDetectionWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def predict(self, img):
        """Predict whether an image is real or fake using CNN Detection"""
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            prob = self.model(tensor).sigmoid().item()
        
        return prob

class FaceForensicsWrapper:
    def __init__(self, model, device, face_extractor=None, is_finetuned=False):
        self.model = model
        self.device = device
        self.face_extractor = face_extractor
        self.is_finetuned = is_finetuned
    
    def predict(self, img):
        """Predict whether an image is real or fake using FaceForensics"""
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Process with face extractor if available
        if self.face_extractor is not None:
            try:
                # Extract face using the face extractor instead of calling detect directly
                faces = self.face_extractor.extract_faces(img_np)
                if faces and len(faces) > 0:
                    # Use the first face found
                    face = faces[0]
                    face = cv2.resize(face, (224, 224))
                else:
                    # If no face detected, use entire image
                    face = cv2.resize(img_np, (224, 224))
            except Exception as e:
                print(f"Error extracting face: {e}")
                # If error, use entire image
                face = cv2.resize(img_np, (224, 224))
        else:
            # If no face extractor, use entire image
            face = cv2.resize(img_np, (224, 224))
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert array to PIL for transform
        pil_face = Image.fromarray(face)
        tensor = transform(pil_face).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.sigmoid(output).item()
        
        # Invert probability for fine-tuned model
        if self.is_finetuned:
            return 1 - prob
            
        return prob

class DummyModel:
    """Dummy model for demonstration purposes"""
    def __init__(self, name):
        self.name = name
    
    def predict(self, img):
        # Return a random prediction
        return np.random.random()

class AIDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Generated Image Detector")
        self.root.geometry("1200x800")
        
        # Create temp directory if needed
        os.makedirs("temp", exist_ok=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Variables
        self.image_path = None
        self.current_image = None
        self.current_face = None
        self.models = {}
        self.status_var = tk.StringVar(value="Ready")
        
        # FaceForensics face detector (initialize once)
        self.face_extractor = None
        if MODELS_AVAILABLE:
            try:
                blazeface = BlazeFace().to(self.device)
                blazeface.load_weights("models/faceforensics_image/blazeface/blazeface.pth")
                blazeface.load_anchors("models/faceforensics_image/blazeface/anchors.npy")
                self.face_extractor = FaceExtractor(facedet=blazeface)
                print("BlazeFace extractor loaded successfully")
            except Exception as e:
                print(f"Error loading BlazeFace extractor: {e}")
        
        # Create frames
        self.create_frames()
        
        # Create widgets
        self.create_widgets()
        
        # Load initial model checkboxes
        self.create_model_checkboxes()
    
    def create_frames(self):
        # Main frames
        self.left_frame = ttk.Frame(self.root, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left frame components
        self.control_frame = ttk.LabelFrame(self.left_frame, text="Controls", padding=10)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        self.model_frame = ttk.LabelFrame(self.left_frame, text="Model Selection", padding=10)
        self.model_frame.pack(fill=tk.X, pady=5)
        
        self.image_frame = ttk.LabelFrame(self.left_frame, text="Uploaded Image", padding=10)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Right frame components
        self.face_frame = ttk.LabelFrame(self.right_frame, text="Detected Face", padding=10)
        self.face_frame.pack(fill=tk.X, pady=5)
        
        self.visualization_frame = ttk.LabelFrame(self.right_frame, text="Results", padding=10)
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Status bar
        self.status_frame = ttk.Frame(self.root, padding=(10, 2))
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_widgets(self):
        # Control frame widgets - combined button for browse and analyze
        self.analyze_button = ttk.Button(self.control_frame, text="Select & Analyze Image", command=self.browse_and_analyze)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        # Image display - fixed size frame
        self.image_container = ttk.Frame(self.image_frame, width=DISPLAY_IMAGE_WIDTH, height=DISPLAY_IMAGE_HEIGHT)
        self.image_container.pack(fill=tk.BOTH, expand=True)
        self.image_container.pack_propagate(False)  # Prevent resizing based on content
        
        self.image_label = ttk.Label(self.image_container)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Face display - fixed size
        self.face_label = ttk.Label(self.face_frame)
        self.face_label.pack(side=tk.LEFT)
        
        # Status bar
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X)
        
    def create_model_checkboxes(self):
        # Clear existing checkboxes
        for widget in self.model_frame.winfo_children():
            widget.destroy()
        
        # Model selection variables
        self.model_vars = {}
        
        # Create a checkbox for each model
        for model_name in MODEL_PATHS.keys():
            var = tk.BooleanVar(value=True)
            self.model_vars[model_name] = var
            checkbox = ttk.Checkbutton(self.model_frame, text=model_name, variable=var)
            checkbox.pack(anchor=tk.W)
    
    def browse_and_analyze(self):
        """Combined function to browse for image and analyze it"""
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(
            title="Select Image to Analyze",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        
        if file_path:
            self.image_path = file_path
            if self.load_image(file_path):
                # Automatically analyze after loading
                self.analyze_image()
    
    def load_image(self, file_path):
        """Load and display image at a fixed size"""
        try:
            # Open image
            self.current_image = Image.open(file_path).convert("RGB")
            
            # Create a copy with fixed dimensions for display
            display_image = self.resize_image_to_fit(self.current_image)
            
            # Convert to Tkinter format
            tk_image = ImageTk.PhotoImage(display_image)
            
            # Update image display
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image  # Keep a reference
            
            # Clear previous results
            self.clear_results()
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            return False
    
    def resize_image_to_fit(self, img):
        """Resize image to fit the fixed dimensions while maintaining aspect ratio"""
        width, height = img.size
        
        # Calculate scaling factors
        width_ratio = DISPLAY_IMAGE_WIDTH / width
        height_ratio = DISPLAY_IMAGE_HEIGHT / height
        
        # Use the smaller ratio to maintain aspect ratio
        ratio = min(width_ratio, height_ratio)
        
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create a new blank image of fixed size
        display_img = Image.new("RGB", (DISPLAY_IMAGE_WIDTH, DISPLAY_IMAGE_HEIGHT), color="white")
        
        # Paste the resized image in the center
        paste_x = (DISPLAY_IMAGE_WIDTH - new_width) // 2
        paste_y = (DISPLAY_IMAGE_HEIGHT - new_height) // 2
        
        display_img.paste(resized_img, (paste_x, paste_y))
        
        return display_img
    
    def clear_results(self):
        """Clear previous results"""
        self.face_label.config(image=None)
        
        # Clear visualization
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()
        
        # Reset status
        self.status_var.set("Ready")
    
    def detect_and_align_face(self):
        """Detect and align face using OpenCV"""
        if self.current_image is None:
            return None
        
        # Convert PIL image to OpenCV format
        img_np = np.array(self.current_image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # If no face detected, use the whole image
        if len(faces) == 0:
            self.current_face = cv2.resize(img_np, (224, 224))
            return self.current_face
        
        # Get the first face
        x, y, w, h = faces[0]
        
        # Add padding around face
        padding = int(w * 0.3)  # 30% padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_np.shape[1], x + w + padding)
        y2 = min(img_np.shape[0], y + h + padding)
        
        # Crop and resize
        face = img_np[y1:y2, x1:x2]
        self.current_face = cv2.resize(face, (224, 224))
        
        return self.current_face
    
    def load_model(self, model_path, model_type):
        """
        Load the appropriate model based on model_type.
        """
        if not MODELS_AVAILABLE:
            self.status_var.set(f"Warning: Using dummy model for {model_type}")
            return DummyModel(model_type)
        
        self.status_var.set(f"Loading {model_type}...")
        self.root.update()
        
        try:
            if "Our Model" in model_type:
                # Load our custom CNN model
                model = FaceClassifierCNN()
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                return OurModelWrapper(model, self.device)
                
            elif "CNN Detection" in model_type:
                # Load CNN Detection model (ResNet50)
                model = resnet50(num_classes=1)
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict['model'])
                model.to(self.device)
                model.eval()
                return CNNDetectionWrapper(model, self.device)
                
            elif "FaceForensics" in model_type:
                # Load FaceForensics model
                is_finetuned = "Fine-tuned" in model_type
                if is_finetuned:
                    # Load fine-tuned model
                    model = getattr(fornet, 'EfficientNetAutoAttB4')().to(self.device)
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                else:
                    # Load pretrained model
                    model = getattr(fornet, 'EfficientNetAutoAttB4')().to(self.device)
                    model_url = weights.weight_url['EfficientNetAutoAttB4_DFDC']
                    model.load_state_dict(load_url(model_url, map_location=self.device, check_hash=True))
                
                model.eval()
                return FaceForensicsWrapper(model, self.device, self.face_extractor, is_finetuned)
                
        except Exception as e:
            self.status_var.set(f"Error loading {model_type}: {str(e)}")
            return DummyModel(model_type)
    
    def analyze_image(self):
        if self.current_image is None:
            messagebox.showinfo("Info", "Please select an image first")
            return
        
        # Clear previous results
        self.clear_results()
        
        # Update status
        self.status_var.set("Detecting face...")
        self.root.update()
        
        # Detect face
        face = self.detect_and_align_face()
        
        if face is None:
            messagebox.showerror("Error", "Failed to process the image")
            return
        
        # Display detected face
        face_img = Image.fromarray(face)
        face_img = face_img.resize((150, 150), Image.LANCZOS)
        tk_face = ImageTk.PhotoImage(face_img)
        self.face_label.config(image=tk_face)
        self.face_label.image = tk_face  # Keep a reference
        
        # Get selected models
        selected_models = {}
        for name, var in self.model_vars.items():
            if var.get():
                selected_models[name] = MODEL_PATHS[name]
        
        if not selected_models:
            messagebox.showinfo("Info", "Please select at least one model")
            return
        
        # Update status
        self.status_var.set("Analyzing image...")
        self.root.update()
        
        # Load models and make predictions
        loaded_models = {}
        results = {}
        
        for name, path in selected_models.items():
            self.status_var.set(f"Loading {name}...")
            self.root.update()
            loaded_models[name] = self.load_model(path, name)
        
        # Process with each model
        face_pil = Image.fromarray(face).convert("RGB")
        
        for name, model in loaded_models.items():
            self.status_var.set(f"Analyzing with {name}...")
            self.root.update()
            
            try:
                # Get prediction
                score = model.predict(face_pil)
                
                # Interpret prediction
                prediction = "Real" if score < 0.5 else "Fake"
                confidence = score if score >= 0.5 else 1 - score
                
                results[name] = {
                    "score": score,
                    "prediction": prediction,
                    "confidence": confidence
                }
            except Exception as e:
                self.status_var.set(f"Error with {name}: {str(e)}")
                results[name] = {"score": None, "prediction": f"Error: {str(e)}"}
        
        # Create visualization
        self.create_visualization(results)
        
        # Display consensus in status bar
        self.display_consensus(results)
    
    def create_visualization(self, results):
        """Create horizontal gauge charts for model predictions"""
        # Clear previous visualization
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()
        
        # Create a frame for the gauges
        gauge_frame = ttk.Frame(self.visualization_frame)
        gauge_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Get models with valid scores
        valid_models = [(name, result) for name, result in results.items() if result["score"] is not None]
        
        if not valid_models:
            # No valid results to show
            no_results_label = ttk.Label(gauge_frame, text="No valid model predictions available")
            no_results_label.pack(pady=20)
            return
        
        # Create figure for all gauges
        fig = plt.figure(figsize=(9, len(valid_models) * 1.2))
        
        # Add horizontal space for labels
        plt.subplots_adjust(left=0.2)
        
        # Create one horizontal gauge for each model
        for i, (name, result) in enumerate(valid_models):
            # Create subplot for this gauge
            ax = fig.add_subplot(len(valid_models), 1, i+1)
            
            # Configure gauge appearance
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_title(name, fontsize=10, loc='left', pad=2)
            
            # Create gauge background - colored sections for real and fake
            ax.axvspan(0, 0.5, color='skyblue', alpha=0.3)  # Real section (0-0.5)
            ax.axvspan(0.5, 1, color='salmon', alpha=0.3)   # Fake section (0.5-1)
            
            # Add text labels for the regions
            ax.text(0.25, 0.5, "REAL", ha='center', va='center', fontsize=9, alpha=0.7)
            ax.text(0.75, 0.5, "FAKE", ha='center', va='center', fontsize=9, alpha=0.7)
            
            # Add decision boundary line at 0.5
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
            
            # Get prediction data
            score = result["score"]
            prediction = result["prediction"]
            confidence = result["confidence"] * 100
            
            # Determine color of the gauge needle based on prediction
            needle_color = 'green' if prediction == "Real" else 'red'
            
            # Draw the gauge needle (a thin triangle/arrow)
            ax.annotate('', xy=(score, 0.5), xytext=(score, 0.2),
                        arrowprops=dict(arrowstyle='->', color=needle_color, lw=2))
            
            # Add a marker at the position
            ax.plot(score, 0.5, 'o', color=needle_color, markersize=8)
            
            # Add confidence text
            confidence_text = f"{prediction} ({confidence:.1f}%)"
            ax.text(score, 0.8, confidence_text, ha='center', fontsize=9,
                    color=needle_color, fontweight='bold')
            
            # Remove most of the frame
            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
                
            # Just keep bottom spine for reference
            ax.spines['bottom'].set_color('gray')
            ax.spines['bottom'].set_alpha(0.5)
            
            # Add tick marks at 0, 0.5, and 1
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['0', '0.5', '1'], fontsize=8)
        
        # Make layout tight
        fig.tight_layout(pad=2)
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=gauge_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add a legend explaining the gauge
        legend_frame = ttk.Frame(self.visualization_frame)
        legend_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        legend_text = "Values below 0.5 indicate a real image, values above 0.5 indicate a fake image."
        legend_label = ttk.Label(legend_frame, text=legend_text, wraplength=400)
        legend_label.pack(pady=5)
    
    def display_consensus(self, results):
        """Display consensus result in status bar"""
        valid_predictions = [r["prediction"] for r in results.values() if r["score"] is not None]
        
        if valid_predictions:
            real_count = valid_predictions.count("Real")
            fake_count = valid_predictions.count("Fake")
            
            if real_count > fake_count:
                consensus = f"Result: Most models ({real_count}/{real_count+fake_count}) predict this is a REAL image"
            else:
                consensus = f"Result: Most models ({fake_count}/{real_count+fake_count}) predict this is a FAKE image"
            
            self.status_var.set(consensus)
        
        if not MODELS_AVAILABLE:
            self.status_var.set("Warning: Real models are not available. Using dummy models that return random predictions.")

# Main function
def main():
    root = tk.Tk()
    
    root.state('zoomed') 

    root.protocol("WM_DELETE_WINDOW", root.quit)
    
    app = AIDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 