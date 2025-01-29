# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:47:32 2025

@author: puran
"""# -*- coding: utf-8 -*-

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
import json
import glob
from tqdm.notebook import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

cwd = os.getcwd()

ROOT_DIR = os.path.join(cwd, "Rashu/train")

labels = {}

for folder in os.listdir(ROOT_DIR):
    print(folder)
    for file in os.listdir(os.path.join(ROOT_DIR, folder)):
        if file.endswith(".jpg"):
            full_name = os.path.join(ROOT_DIR, folder, file)
            labels[full_name] = folder

files = labels.keys()

# We can use different dinov2 version to extract features
# dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14") # 21 M parameters
# dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') # 86 M parameters
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') # 300 M parameters
# dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') # 300 M parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dinov2.to(device)

transform_image = transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Resize((840, 840)), 
                                    # transforms.CenterCrop(224), 
                                    transforms.Normalize([0.5], [0.5])
                                    ])

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    try:
        img = Image.open(img)
        transformed_img = transform_image(img)[:3].unsqueeze(0)
        return transformed_img
    except Exception as e:
        print(f"Error loading image {img}: {e}")
        return None

def compute_embeddings(files: list) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = {}
    
    with torch.no_grad():
      for i, file in enumerate(tqdm(files)):
        embeddings = dinov2(load_image(file).to(device))

        all_embeddings[file] = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()

    with open("all_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    return all_embeddings


# Compute embeddings and store them in a DataFrame
embeddings = compute_embeddings(files)

with open("FinalEmbeddings_train.json", "w") as f:
    f.write(json.dumps(embeddings))

# TRAINING ML model
# from sklearn import svm
# from xgboost import XGBClassifier

# clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, use_label_encoder=False)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.085, max_depth=6, random_state=42)
# clf = HistGradientBoostingClassifier(min_samples_leaf=15, max_iter=100,learning_rate=0.1) # regressor = GradientBoostingRegressor(n_estimators=150, learning_rate=0.01)

y = [labels[file] for file in files]

print(len(embeddings.values()))

embedding_list = list(embeddings.values())

cdd = pd.DataFrame(embedding_list)

# Saving features extracted using dinov2 into excel file
import pandas as pd

print(len(embeddings.values()))

embedding_list = list(embeddings.values())


# Flatten the structure
flattened_data = [embedding_list[i][0] for i in range(len(embedding_list))]

# Convert to DataFrame
df = pd.DataFrame(flattened_data)

# Display the DataFrame shape to verify
print(f"DataFrame shape: {df.shape}")  # Should be (320, 1024)

# Save the DataFrame to an Excel file
df.to_excel("embeddings_output.xlsx", index=False)
print("DataFrame saved to embeddings_output.xlsx")

# Fitting the ML model
clf.fit(np.array(embedding_list).reshape(-1, 1024), y)

from lazypredict.Supervised import LazyClassifier




# TESTING
    
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load and preprocess the test dataset
test_labels = {}  # Dictionary to hold test file paths and their labels
TEST_DIR = os.path.join(cwd, "Rashu/test")

for folder in os.listdir(TEST_DIR):
    for file in os.listdir(os.path.join(TEST_DIR, folder)):
        if file.endswith(".jpg"):
            full_name = os.path.join(TEST_DIR, folder, file)
            test_labels[full_name] = folder

test_files = list(test_labels.keys())

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    try:
        img = Image.open(img)
        transformed_img = transform_image(img)[:3].unsqueeze(0)
        return transformed_img
    except Exception as e:
        print(f"Error loading image {img}: {e}")
        return None

# Extract embeddings for the test dataset
test_embeddings = []
test_y = []

# Extract embeddings with a progress bar
with torch.no_grad():
    for file in tqdm(test_files, desc="Processing Images", unit="image"):
        img_tensor = load_image(file)
        if img_tensor is None:  # Skip corrupted images
            continue
        embedding = dinov2(img_tensor.to(device))
        test_embeddings.append(np.array(embedding[0].cpu()).reshape(1024))  # Flatten to 1D
        test_y.append(test_labels[file])  # Ground truth label
        

# Convert embeddings to NumPy array
# test_embeddings = np.array(test_embeddings)

# Saving dinov2 features into .json
tester = dict(enumerate(test_embeddings.flatten(), 1))
with open("TestEmbeddingsTraining.json", "w") as f:
    f.write(json.dumps(tester))
    
with open("FinalEmbeddingsTesting.json", "w") as f:
    f.write(json.dumps(test_embeddings))
    
    
    
    
import pandas as pd

print(len(test_embeddings.values()))

embedding_list = list(test_embeddings.values())


# Flatten the structure
flattened_data = [test_embeddings[i] for i in range(len(test_embeddings))]

# Convert to DataFrame
df = pd.DataFrame(flattened_data)

# Display the DataFrame shape to verify
print(f"DataFrame shape: {df.shape}")  # Should be (320, 1024)

# Save the DataFrame to an Excel file
df.to_excel("test_embeddings_output.xlsx", index=False)
print("DataFrame saved to embeddings_output.xlsx")

    
for i in range(len(test_embeddings)):
    test_embeddings[i] = list(test_embeddings[i])
        
print(len(test_embeddings.values()))

embedding_list = list(test_embeddings.values())

cdd = pd.DataFrame(embedding_list)

import pandas as pd

# Flatten the structure
flattened_data = [embedding_list[i][0] for i in range(len(embedding_list))]

# Convert to DataFrame
df = pd.DataFrame(flattened_data)

# Display the DataFrame shape to verify
print(f"DataFrame shape: {df.shape}")  # Should be (320, 1024)

# Save the DataFrame to an Excel file
df.to_excel("embeddings_output.xlsx", index=False)
print("DataFrame saved to embeddings_output.xlsx")    
    

# Predict using the trained model
test_predictions = clf.predict(test_embeddings)

# Generate confusion matrix
cm = confusion_matrix(test_y, test_predictions, labels=list(set(test_y)), normalize='true')

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(test_y)))
disp.plot(cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix")
plt.show()

test_y = np.array(test_y)


import pandas as pd

# Read the XLSX file
df = pd.read_excel("FlowerImages.xlsx")

# Convert to CSV 
df.to_csv("output_file.csv", index=False) 

# Training LazyPredict Classifier on our training dataset to check best performing classification model
model = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(np.array(embedding_list).reshape(-1, 1024), test_embeddings, y, test_y)
models

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(np.array(embedding_list).reshape(-1, 1024), y)

y_pred = rf.predict(test_embeddings)

cm = confusion_matrix(test_y, y_pred, labels=list(set(test_y)), normalize='true')

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(test_y)))
disp.plot(cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix")
plt.show()



# Assume `embeddings` is a numpy array of shape (n_samples, embedding_dim)
# Assume `labels` is the target variable
X = np.array([emb for emb in embeddings.values()]).squeeze()  # Convert embeddings dict to array
y = np.array(list(labels.values()))

# Train a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Get feature importance
feature_importance = clf.feature_importances_

l = []
k = []
# Display feature importance
for i, importance in enumerate(feature_importance):
    l.append(importance)
    k.append(i)
    print(f"Feature {i}: {importance}")

import matplotlib.pyplot as plt

# Sort feature importance in descending order
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_importance = feature_importance[sorted_indices]

# Plot the top features
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_importance)), sorted_importance, align='center')
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.title('Feature Importance from Random Forest')
plt.show()

# Get the indices of the top 20 features
top_n = 20
top_indices = sorted_indices[:top_n]

# Plot the top `n` feature importances
plt.figure(figsize=(12, 8))
plt.barh(range(top_n), sorted_importance[:top_n], align='center')
plt.yticks(range(top_n), [f'Feature {i}' for i in top_indices])
plt.xlabel('Feature Importance')
plt.ylabel('Feature Index')
plt.title(f'Top {top_n} Feature Importances')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.show()


import shap
from sklearn.ensemble import RandomForestClassifier

# Train a model
clf = RandomForestClassifier()
clf.fit(X, y)

# Use SHAP to explain the model
explainer = shap.Explainer(clf, X)
shap_values = explainer(X)

# Visualize feature importance
shap.summary_plot(shap_values, X)


# Testing trained ML model
input_file = "ML_Model/test/Flower/Indiangrass_3947430324_1479_1.jpg"

new_image = load_image(input_file)

with torch.no_grad():
    embedding = dinov2(new_image.to(device))

    prediction = clf.predict(np.array(embedding[0].cpu()).reshape(1, -1))

    print("Predicted class: " + prediction[0])


import joblib

# Save the model
joblib.dump(clf, "dinov2_vitl14_=EqualDist.pkl")

# Load the model
loaded_model = joblib.load("trained_model_normalize.pkl")

# Retrieve feature importance from the trained classifier
feature_importance = clf.feature_importances_

# Create a bar plot to visualize feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance, align="center")
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance of DINOv2 Embeddings")
plt.show()


