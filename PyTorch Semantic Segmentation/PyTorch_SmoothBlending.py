# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:25:20 2024

@author: puran
"""

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler
import cv2
import numpy as np
import matplotlib.pyplot as plt

name = 'IMG_5877'

class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # self.loss_fn = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, gamma=3.0)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

# Take a sample image not used while training 
img = cv2.imread("Original/" + name + ".png")
# img = cv2.imread("Orig_Prat/Frame 1.png")


# Also take an original mask for comparison at last
# gt_mask = cv2.imread("Masker/" + name + ".png", 0)
gt_mask = cv2.imread("Final_threshold/" + name + ".png", 0)
# original_mask = cv2.imread("Masks_Prat/Frame 1.png")
# gt_mask = gt_mask[:,:,0]  #Use only single channel...

# foreground = cv2.imread('Foreground masks/'+name + '.jpg', 0)
foreground = cv2.imread('Foreground masks/'+name + '.png', 0)

# Load the trained model
def load_model(model_path):
    model = torch.load(model_path)  # Load the entire model
    model.eval()  # Set the model to evaluation mode
    return model

model_name = 'MAnet_mitb3'
# Load your trained model (Update path as needed)
model = load_model('./'+model_name+'.pth')

# Function to divide image into 256x256 patches
def split_into_patches(image, patch_size=256):
    h, w, _ = image.shape
    patches = []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            # Ensure patch size is exactly patch_size x patch_size (pad if necessary)
            if patch.shape[:2] != (patch_size, patch_size):
                patch = cv2.copyMakeBorder(
                    patch, 0, patch_size - patch.shape[0], 
                    0, patch_size - patch.shape[1], 
                    cv2.BORDER_CONSTANT, value=0
                )
            patches.append(patch)

    return patches, h, w

# Function to merge patches back into original image
def merge_patches(patches, image_height, image_width, patch_size=256):
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.uint8)
    patch_index = 0

    for i in range(0, image_height, patch_size):
        for j in range(0, image_width, patch_size):
            reconstructed_image[i:i+patch_size, j:j+patch_size] = patches[patch_index][:image_height - i, :image_width - j]
            patch_index += 1

    return reconstructed_image

# Function to predict on each patch using the model
def predict_on_patches(model, patches, device='cuda'):
    predictions = []
    model.to(device)

    for patch in patches:
        # Preprocess patch
        patch = patch.transpose(2, 0, 1)  # Convert HWC to CHW
        patch = torch.tensor(patch).float().unsqueeze(0).to(device)  # Add batch dimension

        # Run prediction
        with torch.no_grad():
            output = model(patch)  # Get raw output
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Get class labels
            predictions.append(pred)

    return predictions

# Main function to load image, predict, and reconstruct
def main(image_path, model_path):
    # Load the input image
    image = cv2.imread(image_path)  # Replace with your image path
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split the image into 256x256 patches
    patches, h, w = split_into_patches(image, patch_size=256)

    # Load the trained model
    model = load_model(model_path)

    # Predict on all patches
    predictions = predict_on_patches(model, patches, device='cuda')

    # Merge all patches back into a single image
    final_prediction = merge_patches(predictions, h, w, patch_size=256)

    # Save or display the final predicted image
    cv2.imwrite(name+model_name+'_prediction.png', final_prediction)
    print("Prediction saved as final_prediction.png")

# Run the main function with your image and model paths


main('Original/'+name+'.png', './'+model_name+'.pth')

# image =  cv2.imread(name)
# gt_mask = cv2.imread(name.split('.')[0]+'_mask.png', 0)

print(np.unique(gt_mask))

pr_mask = cv2.imread(name+model_name+'_prediction.png', 0)
# print(np.unique(pr_mask))

k = list(np.unique(pr_mask))
# print(k)

for i in range(1, len(k)):    
    count = np.count_nonzero(pr_mask == k[i])
    # print(count)
    if count <= pr_mask.shape[0]*pr_mask.shape[1]/150:
        pr_mask = np.where(pr_mask == k[i], 0, pr_mask)

k = list(np.unique(pr_mask))
print(k)

# print(name)
print("Actual vegetation cover: ", np.count_nonzero(gt_mask)*100/(gt_mask.shape[0]*gt_mask.shape[1]))
print("Predicted vegetation cover: ", np.count_nonzero(pr_mask)*100/(pr_mask.shape[0]*pr_mask.shape[1]))
print("Foreground vegetation cover: ", np.count_nonzero(foreground)*100/(foreground.shape[0]*foreground.shape[1]))

count = 0
for i in range(1, len(np.unique(gt_mask))):
    count += np.count_nonzero(gt_mask == np.unique(gt_mask)[i])
    
counter = 0
for i in range(1, len(np.unique(pr_mask))):
    counter += np.count_nonzero(pr_mask == np.unique(pr_mask)[i])

plt.figure(figsize=(26,26))
# Original Image
plt.subplot(1, 4, 1)
plt.imshow(img)  # Convert CHW to HWC for plotting
plt.title("Original Image", fontsize=18)
plt.axis("off")

# Foreground Mask
plt.subplot(1, 4, 2)
plt.imshow(foreground, cmap="gray", vmin=0, vmax = 255)  # Visualize ground truth mask
plt.title("Foreground mask", fontsize=18)
plt.axis("off")

# Ground Truth Mask
plt.subplot(1, 4, 3)
plt.imshow(gt_mask, cmap="gray", vmin=0, vmax = np.max(np.unique(gt_mask)))  # Visualize ground truth mask
# plt.xlabel('Bermuda')
# plt.imshow(gt_mask, cmap="gray", vmin=0, vmax = 8)  # Visualize ground truth mask
plt.title("Ground truth mask", fontsize=18)
plt.axis("off")

# Predicted Mask
plt.subplot(1, 4, 4)
# plt.imshow(pr_mask, cmap="gray", vmin=0, vmax = 8)  # Visualize predicted mask
plt.imshow(pr_mask, cmap="gray", vmin=0, vmax = np.max(np.unique(pr_mask)))  # Visualize predicted mask
plt.title("Predicted mask", fontsize=18)
plt.axis("off")

# Show the figure
plt.show()
