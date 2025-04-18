import os
from PIL import Image
import glob

# Define the path to the original dataset
original_dataset_path = 'trashnet/'

# Define the path to save the resized dataset
resized_dataset_path = 'trashnet-1'

# Create the resized dataset directory if it doesn’t exist
os.makedirs(resized_dataset_path, exist_ok=True)

# Define the target size (224x224 pixels)
target_size = (224, 224)

# Iterate through each class folder in the original dataset
for class_folder in glob.glob(os.path.join(original_dataset_path, '*')):
    if os.path.isdir(class_folder):
        # Extract the class name from the folder path
        class_name = os.path.basename(class_folder)
        # Create a corresponding folder in the resized dataset directory
        resized_class_folder = os.path.join(resized_dataset_path, class_name)
        os.makedirs(resized_class_folder, exist_ok=True)
        
        # Iterate through each image in the class folder
        for image_path in glob.glob(os.path.join(class_folder, '*')):
            # Check if the file is an image (supports common formats)
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Open the image
                with Image.open(image_path) as img:
                    # Resize the image to 224x224 with Lanczos resampling
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    # Define the path to save the resized image
                    resized_image_path = os.path.join(resized_class_folder, os.path.basename(image_path))
                    # Save the resized image
                    resized_img.save(resized_image_path)

print("Dataset resized to 224x224 and saved to", resized_dataset_path)