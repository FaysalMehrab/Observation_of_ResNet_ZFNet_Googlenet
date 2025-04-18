import os
import torch
import torch.nn as nn
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

# Define paths
dataset_path = os.path.normpath('Trashnet')  # Adjust to your dataset path
output_path = os.path.normpath('trashnet_features_zfnet')  # ZFNet features
os.makedirs(output_path, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define ZFNet architecture
class ZFNet(nn.Module):
    def __init__(self):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.AdaptiveAvgPool2d((6, 6)),  # Ensure 6x6 output
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),  # 9216 = 256 * 6 * 6
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Stop before final classification layer
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# Load ZFNet
try:
    model = ZFNet()
    model.eval()
    model.to(device)
    print("ZFNet loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load ZFNet: {e}")

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