# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 04:51:29 2024

@author: puran
"""

from ultralytics import YOLO

#Instance
# model = YOLO('yolov8x-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8x.pt')  # Transfer the weights from a pretrained model (recommended for training)

# define number of classes based on YAML
import yaml
with open("data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])
    
project = "Results/"

#Define subdirectory for this specific training
name = "Yolov8x-new" 

# Train the model
results = model.train(data='data.yaml',
                      project=project,
                      name=name,
                      # optimizer='Adam',            
                      epochs=100,
                      # patience=0,
                      batch=16,
                      imgsz=512,
                      # amp=False,
                      # lr0=0.0001,
                      # optimize=True,
                      # autoaugment='randaugment',
                      # pretrained=True,
                      # cache=True,
                      dropout=0)


bluebie = YOLO("Blueberry/Extra/Results/Yolov8x-new2/weights/best.pt")

new_image = 'Blueberry/14-323R1/'
new_results = bluebie.predict(new_image, conf=0.6)  #Adjust conf threshold

# View results
# for r in new_results:
#     print(r.boxes)  # print the Boxes object containing the detection bounding boxes

import matplotlib.pyplot as plt

new_result_array = new_results[5].plot()
plt.figure(figsize=(12, 12))
plt.imshow(new_result_array)