# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:58:31 2024

@author: puran
"""

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load DINOv2 model
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')  # Large model
dinov2.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2.to(device)

# Define image preprocessing
transform_image = transforms.Compose([
    transforms.Resize((840, 840)),  # Resize image
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize (DINOv2 standard)
])

def load_image(img_path):
    """
    Load an image and apply the preprocessing.
    """
    img = Image.open(img_path).convert("RGB")
    return transform_image(img).unsqueeze(0).to(device)  # Add batch dimension

def extract_and_visualize_embedding(model, img_path, embedding_dim):
    """
    Extract embeddings and visualize the contribution of a specific embedding value.
    Args:
        model: Pre-trained DINOv2 model
        img_path: Path to the input image
        embedding_dim: Index of the embedding dimension to visualize
    """
    # Load and preprocess the image
    img_tensor = load_image(img_path)

    # Forward pass to get embeddings
    with torch.no_grad():
        outputs = model.get_intermediate_layers(img_tensor, n=1)[0]  # Extract patch embeddings
    
    # Extract patch embeddings (exclude [CLS] token)
    patch_embeddings = outputs[:, 1:, :]  # Shape: [1, num_patches, 1024]

    # Check the shape of patch embeddings
    print(f"Patch Embeddings Shape: {patch_embeddings.shape} (num_patches x embedding_dim)")

    # Extract the specific embedding dimension across all patches
    embedding_values = patch_embeddings[0, :, embedding_dim]  # Shape: [num_patches]

    # Dynamically determine patch grid size
    num_patches = embedding_values.shape[0]
    grid_height = int(np.sqrt(num_patches))
    grid_width = num_patches // grid_height
    print(f"Grid Size: {grid_height} x {grid_width}")

    # Reshape embedding values back to a grid
    embedding_map = embedding_values.cpu().numpy().reshape(grid_height, grid_width)

    # Normalize embedding map for visualization
    embedding_map_resized = np.interp(embedding_map, (embedding_map.min(), embedding_map.max()), (0, 1))
    embedding_map_resized = Image.fromarray((embedding_map_resized * 255).astype(np.uint8)).resize((840, 840))

    # Load the original image for overlay
    original_image = Image.open(img_path).convert("RGB").resize((840, 840))

    # Plot the original image and the embedding heatmap overlay
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")
    
    # Overlay embedding heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)
    plt.imshow(embedding_map_resized, cmap="jet", alpha=0.5)  # Heatmap overlay
    plt.title(f"Embedding Dimension #{embedding_dim} Heatmap")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# Path to the image
img_path = "Finale/Training/Flower/Big Bluestem_1584420435_7_1.jpg"  # Replace with your image path

# Extract embeddings and visualize the 25th dimension
extract_and_visualize_embedding(dinov2, img_path, embedding_dim=25)