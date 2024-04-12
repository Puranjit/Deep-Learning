# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 09:53:38 2023

@author: puran
"""

import cv2
import numpy as np
import os

img_dir = "train/images/"
count = 0
x = []
y = []

for path, subdirs, files in os.walk(img_dir):
    # print(path)  
    dirname = path.split(os.path.sep)[-1]
    # print(subdirs)
    images = os.listdir(path)  #List of all image names in this subdirectory
    # print(images)
    for i, image_name in enumerate(images):

    # Read the image using OpenCV
        if image_name.endswith(".jpg"):
            
            count+=1
            
            # file = open(dirname+image_name, 'r')
            # content = file.read()
            # break
        
            # with open(image_name, 'r') as file:
            #     # Read all lines from the file
            #     lines = file.readlines()

            # break
        
            image = cv2.imread(img_dir+'/'+image_name)
            x.append(image.shape[0])
            y.append(image.shape[1])
            
            # if (np.unique(image) != 0):
            #     cv2.imwrite("train/BlueBie/"+image_name+".png", new_image)
            # Get the current size of the image
            # original_height, original_width, _ = image.shape
            
            # # padding_right = 224*((original_width//224)+1)-original_width
            # # padding_bottom = 224*((original_height//224)+1)-original_height
            
            # padding_bottom = 4032 - original_height
            # padding_right = 3136 - original_width
        
            # # Calculate the new size with padding
            # new_width = original_width + padding_right
            # new_height = original_height + padding_bottom
        
            # # Use np.pad to add zero padding to the right and bottom
            # new_image = np.pad(image, ((0, padding_bottom), (0, padding_right), (0, 0)), mode='constant', constant_values=0)
        
            # Save the new image with padding
            cv2.imwrite("train/BBery/"+image_name+".png", new_image)

# Example usage
# input_image_path = "Star1-S2_jpg.rf.f0f29450f453ca6784e5ea365366ba7c.jpg"  # Replace with the path to your input image
# output_image_path = "Star1S2_pad.png"  # Replace with the desired output path
# # padding_right = 100  # Adjust the width of the padding to the right
# # padding_bottom = 50  # Adjust the height of the padding at the bottom
# add_zero_padding(input_image_path, output_image_path)

