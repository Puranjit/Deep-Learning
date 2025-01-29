# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:24:56 2024

@author: puran
"""

from dinov2.models import build_dinov2_vitl14

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((768, 768)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet stats)
])

# Load train and validation datasets
train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(root='dataset/val', transform=transform)

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

from dinov2.models import build_dinov2_vitl14
import torch.nn as nn

# Load Dinov2 pretrained model
model = build_dinov2_vitl14(pretrained=True)

# Update classification head for 2 classes
num_classes = 2
model.head = nn.Linear(model.embed_dim, num_classes)

for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True
    
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), 'dinov2_binary_classification.pth')