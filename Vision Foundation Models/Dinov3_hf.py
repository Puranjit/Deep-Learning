# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:26:25 2026

@author: puran
"""

import torch
import os
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import login

# === Load DINOv3 ===
print("Loading DINOv3...")

# === Authenticate with HuggingFace ===
login(token="###")

# Available DINOv3 ViT models:
# vits16  → small   (384 dim)
# vitb16  → base    (768 dim)   ← recommended starting point
# vitl16  → large   (1024 dim)
# vitg16  → giant   (1536 dim)
# vit7b16 → 7B      (very large)

model = "vith16plus"  # change to vits16, vitl16, vitg16 as needed
model_name = f"facebook/dinov3-{model}-pretrain-lvd1689m"

dino_processor = AutoImageProcessor.from_pretrained(model_name)
dino_model = AutoModel.from_pretrained(model_name)
dino_model.eval()

# === Process all images ===
year = 2015
results = []

for k in range(6):
    year += 1
    for i in range(1, 13):
    # for i in range(9, 13):
        month_str = f"0{i}" if i < 10 else str(i)
        image_dir = f"Original/{year}/{month_str}/"

        if not os.path.exists(image_dir):
            continue

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for filename in image_files:
            print(f"Processing: {filename}")
            image_path = os.path.join(image_dir, filename)

            # === DINOv3 CLS embedding ===
            image_pil = Image.open(image_path).convert("RGB")
            dino_inputs = dino_processor(images=image_pil, return_tensors="pt")

            with torch.no_grad():
                dino_outputs = dino_model(**dino_inputs)
                last_hidden_state = dino_outputs.last_hidden_state
                # Token layout: [CLS, REG_1, ..., REG_n, patch_1, ..., patch_N]
                cls_embedding = last_hidden_state[:, 0, :].numpy().flatten()

            row = {'filename': filename}
            for idx, val in enumerate(cls_embedding):
                row[f'dino_cls_{idx}'] = val

            results.append(row)

    # === Save to CSV per year ===
    output_dir = f"Original/DINOv3_{model}/"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_excel(f"{output_dir}{year}_DINOv3_{model}_embeddings.xlsx", index=False)
    