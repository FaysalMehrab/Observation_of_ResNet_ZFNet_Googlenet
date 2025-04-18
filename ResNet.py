import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import numpy as np

# Verify NumPy
try:
    np.zeros(1)
    print("NumPy is available.")
except ImportError as e:
    raise ImportError("NumPy is not installed. Run 'pip install numpy==1.26.4'.")

# Define paths (normalized for Windows)
dataset_path = os.path.normpath('Trashnet')  # Adjust to 'Trashnet' if needed
output_path = os.path.normpath('trashnet_features_resnet')
os.makedirs(output_path, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load ResNet-101
try:
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)
    print("ResNet-101 loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load ResNet-101: {e}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract features function
def extract_features(image_path, model, transform):
    try:
        # Verify image
        with Image.open(image_path) as img:
            img.verify()
        # Process image
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            image = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model(image)
                features = features.squeeze()
                if features.numel() == 0:
                    raise ValueError("Empty feature tensor.")
                return features.cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Process dataset
for class_folder in glob.glob(os.path.join(dataset_path, '*')):
    if os.path.isdir(class_folder):
        class_name = os.path.basename(class_folder)
        print(f"Processing class: {class_name}")
        class_output_path = os.path.normpath(os.path.join(output_path, class_name))
        os.makedirs(class_output_path, exist_ok=True)

        for image_path in glob.glob(os.path.join(class_folder, '*')):
            image_path = os.path.normpath(image_path)
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                features = extract_features(image_path, model, transform)
                if features is not None:
                    feature_filename = os.path.splitext(os.path.basename(image_path))[0] + '.npy'
                    feature_path = os.path.normpath(os.path.join(class_output_path, feature_filename))
                    try:
                        np.save(feature_path, features)
                        print(f"Saved features for {image_path}")
                    except Exception as e:
                        print(f"Error saving features for {image_path}: {str(e)}")

print("Feature extraction completed. Features saved to", output_path)