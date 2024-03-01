# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:15:05 2023

@author: puran
"""

from ultralytics import YOLO

#Instance
# model = YOLO('yolov8x-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8x-obb.pt')  # Transfer the weights from a pretrained model (recommended for training)

# define number of classes based on YAML
import yaml
with open("data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])
    
project = "Results/"

#Define subdirectory for this specific training
name = "Yolov8x-obb" 

# Train the model
results = model.train(data='data.yaml',
                      project=project,
                      name=name,
                      optimizer='Adam',            
                      epochs=100,
                      # patience=0,
                      batch=32,
                      imgsz=224,
                      # amp=False,
                      # lr0=0.0001,
                      # optimize=True,
                      # autoaugment='randaugment',
                      # pretrained=True,
                      # cache=True,
                      dropout=0)