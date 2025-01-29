# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:47:00 2024

@author: puran
"""
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def extract_dinov2_features(image_path, model, transform):
    """
    Extract features for a single image using DINOv2 with global average pooling.
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        if 'x_norm_patchtokens' in features:
            patch_features = features['x_norm_patchtokens'].squeeze(0)  # Shape: [num_patches, 1024]
            features = patch_features.mean(dim=0).cpu().numpy()  # Global average pooling → Shape: [1024]
        else:
            raise KeyError("Expected feature key not found")
    
    return features

def prepare_dataset(root_dir, model, transform):
    """
    Prepare features and labels from a directory structure
    Assumes directory structure:
    root_dir/
    ├── Flower/
    └── Non-Flower/
    """
    X = []
    y = []
    
    # Process Flower images
    flower_dir = os.path.join(root_dir, 'Flower')
    for filename in os.listdir(flower_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(flower_dir, filename)
            features = extract_dinov2_features(image_path, model, transform)
            X.append(features)
            y.append(1)  # 1 for Flower
    
    # Process Non-Flower images
    non_flower_dir = os.path.join(root_dir, 'Non Flower')
    for filename in os.listdir(non_flower_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(non_flower_dir, filename)
            features = extract_dinov2_features(image_path, model, transform)
            X.append(features)
            y.append(0)  # 0 for Non-Flower
    
    return np.array(X), np.array(y)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)

# Transformation (same as in your original code)
transform_image = transforms.Compose([
    transforms.Resize((840, 840)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Prepare dataset
root_dir = 'ML Model'  # Adjust this to your dataset path
X, y = prepare_dataset(root_dir, dinov2, transform_image)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you can compute feature importance
from sklearn.ensemble import RandomForestClassifier

# Compute feature importance
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Get feature importance
feature_importance = rf_classifier.feature_importances_

# Assume you already have a trained Dinov2 model loaded
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

# Load and preprocess the image
transform_image = transforms.Compose([
    transforms.Resize((840, 840)),  # Rescale image to input size
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return transform_image(img).unsqueeze(0)  # Add batch dimension

# Extract embeddings and intermediate feature maps
def get_feature_maps(image_path, model):
    global features
    img_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        features = model.forward_features(img_tensor)  # Extract feature maps
    return features

def visualize_top_features(image_path, feature_maps, feature_importance, top_n):
    # Extract the correct feature map tensor
    global importance_sorted_indices
    if 'x_norm_patchtokens' in feature_maps:
        spatial_features = feature_maps['x_norm_patchtokens'].squeeze(0).cpu().numpy()  # Shape: [n_patches, embedding_dim]
    else:
        raise KeyError("The expected key 'x_norm_patchtokens' is not found in feature_maps.")
    
    # Get the top N features based on importance
    # importance_sorted_indices = np.argsort(feature_importance)[::-1][:top_n]
    importance_sorted_indices = np.argsort(feature_importance)[::-1][top_n:top_n+100]

    # Plot the original image
    original_img = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(original_img)
    plt.axis('off')
    plt.title("Original Image")
    plt.show()

    # Ensure spatial_features has dimensions [n_patches, H, W]
    # You might need to reshape or rearrange based on the feature shape
    num_patches = spatial_features.shape[0]
    h = w = int(num_patches**0.5)  # Assuming a square grid of patches
    reshaped_features = spatial_features.reshape(h, w, -1)  # Reshape to [H, W, embedding_dim]

    # Plot the top features
    fig, axes = plt.subplots(x, x, figsize=(120,120))  # Adjust grid size for top 10 features (2 rows, 5 columns)
    for i, idx in enumerate(importance_sorted_indices):
        ax = axes[i // x, i % x]
        feature_map = reshaped_features[:, :, idx]  # Extract the selected feature
        ax.imshow(feature_map, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Feature {idx}", fontsize=45)
    plt.tight_layout()
    if np.max(feature_importance) == feature_importance[importance_sorted_indices[0]]:
        plt.savefig(+image_path.split('/')[3].split('.')[0]+'_1_100'+'.png')
    else:
        plt.savefig(+image_path.split('/')[3].split('.')[0]+'_101_200'+'.png')
    plt.show()
    
# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2.to(device)
image_path = "Finale/Sub_Test/Flower/Big Bluestem_2868222639_631_1.jpg"

feature_maps = get_feature_maps(image_path, dinov2)

# Use your computed feature importance from Random Forest or other methods
# feature_importance = np.random.rand(1024)  # Example: Replace with actual importance scores

x = 10

# Visualize top 10 features
visualize_top_features(image_path, feature_maps, feature_importance, top_n=x**2)
