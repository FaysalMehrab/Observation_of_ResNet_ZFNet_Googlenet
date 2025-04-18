import os
import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define paths
nn_folder = os.path.normpath('NN')  # Folder with class subfolders containing one image each
features_path = os.path.normpath('trashnet_features_resnet')  # Folder with .npy features

# Verify NumPy
try:
    np.zeros(1)
    print("NumPy is available.")
except ImportError as e:
    raise ImportError("NumPy is not installed. Run 'pip install numpy==1.26.4'.")

# Check if NN folder exists
if not os.path.exists(nn_folder):
    raise FileNotFoundError(f"NN folder does not exist at {nn_folder}")

# Collect all feature files and their paths
all_features = []
all_feature_paths = []

# Iterate through all class subfolders in trashnet_features_resnet
for class_folder in glob.glob(os.path.join(features_path, '*')):
    if os.path.isdir(class_folder):
        class_name = os.path.basename(class_folder)
        for feature_file in glob.glob(os.path.join(class_folder, '*.npy')):
            feature_path = os.path.normpath(feature_file)
            try:
                features = np.load(feature_file)
                all_features.append(features)
                all_feature_paths.append(feature_path)
                print(f"Loaded features from {feature_path}")
            except Exception as e:
                print(f"Error loading {feature_path}: {str(e)}")

# Convert features to numpy array
all_features = np.array(all_features)
print(f"Total features loaded: {len(all_features)}")

# Process subfolders in NN folder
for nn_subfolder in glob.glob(os.path.join(nn_folder, '*')):
    if os.path.isdir(nn_subfolder):
        class_name = os.path.basename(nn_subfolder)
        print(f"\nProcessing class subfolder: {class_name}")

        # Look for images in the subfolder
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif')
        nn_images = []
        for ext in image_extensions:
            nn_images.extend(glob.glob(os.path.join(nn_subfolder, ext)))

        if not nn_images:
            print(f"No images found in {nn_subfolder}")
            continue

        # Take the first image (assuming one image per subfolder)
        nn_image_path = nn_images[0]
        nn_image_name = os.path.splitext(os.path.basename(nn_image_path))[0]
        print(f"Selected image: {nn_image_name}")

        # Find the corresponding feature file
        feature_file = os.path.normpath(os.path.join(features_path, class_name, f"{nn_image_name}.npy"))

        # Load the features of the NN image
        if os.path.exists(feature_file):
            try:
                query_features = np.load(feature_file)
                print(f"Loaded features for {nn_image_path} from {feature_file}")
            except Exception as e:
                print(f"Error loading features from {feature_file}: {str(e)}")
                continue
        else:
            print(f"Feature file not found for {nn_image_path}: {feature_file}")
            continue

        # Compute cosine similarity
        similarities = cosine_similarity(query_features.reshape(1, -1), all_features)[0]

        # Get indices of top 10 neighbors (excluding the image itself)
        neighbor_indices = np.argsort(similarities)[::-1]
        neighbor_count = 0
        top_neighbors = []

        for idx in neighbor_indices:
            if len(top_neighbors) >= 10:
                break
            # Skip the query image itself
            if all_feature_paths[idx] != feature_file:
                top_neighbors.append((all_feature_paths[idx], similarities[idx]))
                neighbor_count += 1

        # Print results
        print(f"Top 10 nearest neighbors for {os.path.basename(nn_image_path)} (class: {class_name}):")
        for i, (neighbor_path, similarity) in enumerate(top_neighbors, 1):
            neighbor_class = os.path.basename(os.path.dirname(neighbor_path))
            neighbor_image = os.path.splitext(os.path.basename(neighbor_path))[0]
            print(f"{i}. {neighbor_image} (class: {neighbor_class}, similarity: {similarity:.4f})")

print("\nNearest neighbor search completed.")