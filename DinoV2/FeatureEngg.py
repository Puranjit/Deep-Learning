# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:53:08 2024

@author: puran
"""


def visualize_top_features(X_train, feature_importance, top_n=25):
    """
    Visualize top features based on their importance from the training dataset
    
    Parameters:
    - X_train: Training features extracted by DINOv2
    - feature_importance: Importance scores of features
    - top_n: Number of top features to visualize
    """
    # Ensure feature_importance matches X_train feature dimension
    print("X_train shape:", X_train.shape)
    print("Feature importance shape:", feature_importance.shape)
    
    # Sort feature importance and get top N indices
    top_feature_indices = np.argsort(feature_importance)[::-1][:top_n]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    # fig.suptitle('Top 25 Most Important Features', fontsize=16)
    
    # Flatten the axes for easy indexing
    axes = axes.flatten()
    
    # Iterate through top features
    for i, feature_idx in enumerate(top_feature_indices):
        # Create a 2D representation of the feature
        # Use the original feature values across all training samples
        feature_values = X_train[:, feature_idx]
        
        # Compute a 2D representation
        # Option 1: Reshape to a square if possible
        try:
            feature_dim = int(np.sqrt(len(feature_values)))
            feature_2d = feature_values.reshape(feature_dim, feature_dim)
        except ValueError:
            # Option 2: If reshaping fails, create a 2D heatmap using the first few samples
            feature_2d = feature_values[:100].reshape(10, 10)
        
        # Plot the feature
        im = axes[i].imshow(feature_2d, cmap='viridis', aspect='auto')
        axes[i].set_title(f'Feature {feature_idx}\nImportance: {feature_importance[feature_idx]:.4f}')
        axes[i].axis('off')
        
        # Add a colorbar to each subplot
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    # Remove any extra subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    plt.show()
    
print("X_train shape:", X_train.shape)
print("Feature importance shape:", feature_importance.shape)

# Now you can use this in your visualization function
visualize_top_features(X_train, feature_importance)
