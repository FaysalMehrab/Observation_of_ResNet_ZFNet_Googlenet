import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import numpy as np
import torch.nn as nn

# Define paths
dataset_path = 'Trashnet'  # Your dataset folder with class subfolders
output_path = 'trashnet_features_googlenet'  # Where to save GoogLeNet features
os.makedirs(output_path, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pre-trained GoogLeNet
model = models.googlenet(pretrained=True)
model.fc = nn.Identity()  # Remove the final classification layer to get 1024-dim features
model.eval()
model.to(device)
print("Pre-trained GoogLeNet loaded successfully.")

# Define transformations (standard for GoogLeNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract features
def extract_features(image_path, model, transform):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            image = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model(image)
                return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Process the dataset
for class_folder in glob.glob(os.path.join(dataset_path, '*')):
    if os.path.isdir(class_folder):
        class_name = os.path.basename(class_folder)
        print(f"Processing class: {class_name}")
        class_output_path = os.path.join(output_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        for image_path in glob.glob(os.path.join(class_folder, '*')):
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                features = extract_features(image_path, model, transform)
                if features is not None:
                    feature_filename = os.path.splitext(os.path.basename(image_path))[0] + '.npy'
                    feature_path = os.path.join(class_output_path, feature_filename)
                    np.save(feature_path, features)
                    print(f"Saved features for {image_path}")

print("Feature extraction completed. Features saved to", output_path)