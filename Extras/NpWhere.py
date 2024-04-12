# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:20:54 2023

@author: pzs0098
"""

import os
import cv2

from PIL import Image as im
import numpy as np

# img_dir = "Dog/train/labels"
img_dir = "Blueberry/"
count = 0

for path, subdirs, files in os.walk(img_dir):
    # print(path)  
    dirname = path.split(os.path.sep)[-1]
    # print(subdirs)
    images = os.listdir(path)  #List of all image names in this subdirectory
    # print(images)
    for i, image_name in enumerate(images):
        if image_name.endswith("6 S2.jpg"):

            # print(image_name.split('.txt')[0])
            # if image_name.split('.png')[0] == "":

            original_image = cv2.imread(img_dir+'/'+image_name)


            # if len(np.unique(original_image)) > 3:
            #     cv2.imwrite('Bii/train/images/BBery/'+image_name.split('.jpg')[0]+'.png', original_image)  # Save each sub-image to a file

            # count += 1
            # if count == 11:
            #     break

            # Define the size of sub-images
            sub_image_size = (224, 224)
            
            # Calculate the number of rows and columns for sub-images
            num_rows = original_image.shape[0] // sub_image_size[0]
            num_cols = original_image.shape[1] // sub_image_size[1]
            
            # Create a list to store sub-images
            sub_images = []
            # Iterate through the image and extract sub-images
            for i in range(num_rows):
                for j in range(num_cols):
                    left = j * sub_image_size[1]
                    upper = i * sub_image_size[0]
                    right = (j + 1) * sub_image_size[1]
                    lower = (i + 1) * sub_image_size[0]
                    sub_img = original_image[upper:lower, left:right]
                    x = np.unique(sub_img)
                    if x.mean() == 0:
                        continue
                    else:
                        sub_images.append(sub_img)
            
            # Save the sub-images or perform other operations on them
            for i, sub_img in enumerate(sub_images):
                cv2.imwrite('Blueberry/Star6 S2/'+image_name+f'{i}.png', sub_img)  # Save each sub-image to a file
            # break
                 
            # # Get the current size of the image2
            # padding_right = 224*((original_width//224)+1)-original_width
            # padding_bottom = 224*((original_height//224)+1)-original_height

            # # Calculate the new size with padding
            # new_width = original_width + padding_right
            # new_height = original_height + padding_bottom

            # # Use np.pad to add zero padding to the right and bottom
            # new_image = np.pad(original_image, ((0, padding_bottom), (0, padding_right), (0, 0)), mode='constant', constant_values=0)

            # # Save the new image with padding
            # # cv2.imwrite(output_path, new_image)
            # count+=1
            # cv2.imwrite(img_dir+'/'+image_name, new_image)

            # original_image = im.open(img_dir+'/'+image_name)  # Replace 'your_image.jpg' with the path to your image            
                       
            # count+=1
            # print(i, image_name)
            # if i == 37:
            # print(image_name)
            # count+=1

            img = cv2.imread(img_dir+'/'+image_name)
            
            
            for i in range(0,224,224):
                print(i)
            
            # if count == 50:
            #     break
            
            # k = list(np.unique(img))
            
            # # k  = [0,1,2,3,8,9,10,12,32]
            
            # for n in k:

            #     if n >= 11:
            #         count+=1
            #         break
            
            # if count==1:
            #     break
            
            print(image_name,' ' ,np.unique(img))
        
            # img = np.where(img == 1, 11, img)
            # img = np.where(img == 5, 10, img)
            # img = np.where(img == 8, 5, img)
            # img = np.where(img == 6, 8, img)
            # img = np.where(img == 9, 6, img)
            # img = np.where(img == 4, 9, img)
            
            # img = np.where(img == 2, 12, img) 
            # img = np.where(img == 7, 2, img)
            # img = np.where(img == 3, 7, img)
            
            
            
            
            # img = np.where(img == 12, 4, img)
            
            # cv2.imwrite(img_dir+'/'+image_name, img)
            
            # for i in range(len(k)):
            #     count = np.count_nonzero(img == k[i])
            #     if count < 25000:
            #         img = np.where(img == k[i], 0, img)
            #         cv2.imwrite(img_dir+'/'+image_name, img)
            #         # print('Changes made in the image', image_name)
            #     else:
            #         print('No changes made in the image', image_name)
            
            
            # if count == 21:
            #     break
            # # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # # k.append(img)
            # # if (np.unique(img)[-1] == 8):
            # #     k = img
                
            # print(np.unique(img))
            
            
            # # img = np.where(img==1,2,img)
            # # break

            
            # # cv2.imwrite(img_dir+'/'+image_name, gray)
            # # print(np.unique(img)[0:, len(np.unique(img))])
            # # break
            # # print(np.unique(img))
            # # break
            
            # image = im.open(path+'/'+image_name)
            # re_image = image.resize((2048, 2048))
            # newpath = path+'/Cropped Res/'+image_name            
            # re_image.save(newpath)
            
# img = cv2.imread('Aug 3/DSC_8449.png')
# k = list(np.unique(img))
# addi = 0
# for i in range(len(k)):
#     value = k[i]
#     count = np.count_nonzero(img == value)
#     print("The value ", value, " appears", count, "times in the array.")
#     addi += count    
# addi    

# img = cv2.imread('July 8/J_IV_13.png')
# np.unique(img)
# img = np.where(img == 7, 6, img)
# cv2.imwrite('July 8/J_IV_13.png', img)

# cv2.imwrite(img_dir+'/'+image_name, img)
# count = np.count_nonzero(img == 4)
