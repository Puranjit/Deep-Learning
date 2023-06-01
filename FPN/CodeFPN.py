# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:54:49 2023

@author: pzs0098
"""

# importing libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
from keras.models import load_model
import random
# import time

# start_time = time.time()

# Reading Images from the training dataset
train_img_path = "Data_new/train_images/train/" # Change
train_mask_path = "Data_new/train_masks/train/" # Change


img_list = os.listdir(train_img_path)
msk_list = os.listdir(train_mask_path)

num_images = len(os.listdir(train_img_path))

# Defining batch_size and total number of pixel associated classes in our annotated dataset
seed = 24
batch_size = 12
n_classes = 12

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from tensorflow.keras.utils import to_categorical

# Defining backbone
# available models that could be used as backbone: ['resnet18', 'resnet34', 'resnet50', 
# 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 
# 'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101', 
# 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'inceptionresnetv2', 
# 'inceptionv3', 'mobilenet', 'mobilenetv2', 'efficientnetb0', 'efficientnetb1', 
# 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 
# 'efficientnetb7']

# Use this to preprocess input for transfer learning
BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)

#Define a function to perform additional preprocessing after datagen.
#For example, scale images, convert masks to categorical, etc. 
def preprocess_data(img, mask, n_classes):
    #Scale images
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    # print('Kidda')
    # img = preprocess_input(img)  #Preprocess based on the pretrained backbone...
    #Convert mask to one-hot
    # mask = np.where(mask==117,1,mask)
    # mask = np.where(mask==199,2,mask)
    # print(np.unique(mask))
    mask = to_categorical(mask, n_classes)
    
    # print('HHOO')
      
    return (img,mask)

#Define the generator.
#We are not doing any rotation or zoom to make sure mask values are not interpolated.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def trainGenerator(train_img_path, train_mask_path, num_class):
    
    img_data_gen_args = dict(horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')
    
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)
    
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        # target_size = (240,240),
        # color_mode='rgb',
        # interpolation='nearest',
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        # target_size = (240,240),
        # interpolation='nearest',
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    # print('hello')
        
    for (img, mask) in train_generator:
    
        # k = list(np.unique(mask))
        
        # for i in range(len(k)):
        #     count = np.count_nonzero(mask == k[i])
        #     if count < 5000:
        #         mask = np.where(mask == k[i], 0, mask)
                
        img, mask = preprocess_data(img, mask, num_class)
        # print('hi')
        yield (img, mask)

# Run the code to perform data augmentation on train dataset
train_img_path = "Data_new/train_images/" # Change
train_mask_path = "Data_new/train_masks/" # Change
train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class = 12)

# Run the code to perform data augmentation on train dataset
val_img_path = "Data_new/val_images/" # Change
val_mask_path = "Data_new/val_masks/" # Change
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class = 12)

x, y = train_img_gen.__next__()

# for i in range(0,3):
#     image = x[i]
#     mask = np.argmax(y[i], axis=2)
#     plt.subplot(1,2,1)
#     plt.imshow(image)
#     plt.subplot(1,2,2)
#     plt.imshow(mask, cmap='gray')
#     plt.show()

x_val, y_val = val_img_gen.__next__()

# for i in range(0,3):
#     image = x_val[i]
#     mask = np.argmax(y_val[i], axis=2)
#     plt.subplot(1,2,1)
#     plt.imshow(image)
#     plt.subplot(1,2,2)
#     plt.imshow(mask, cmap='gray')
#     plt.show()


num_train_imgs = len(os.listdir('Data_new/train_images/train/')) # Change
num_val_images = len(os.listdir('Data_new/val_images/val/')) # Change

# Defining steps_per_epoch that will be used in an epoch while training, and validation
steps_per_epoch = num_train_imgs//batch_size+1
val_steps_per_epoch = num_val_images//batch_size+1

IMG_HEIGHT = x.shape[1]
IMG_WIDTH  = x.shape[2]
IMG_CHANNELS = x.shape[3]

# The following two lines helps to set a framework
sm.set_framework('tf.keras')
sm.framework()

# Defining metrics
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), 
           sm.metrics.Precision(), sm.metrics.Recall()]

model = sm.FPN(BACKBONE, encoder_weights='imagenet',
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                classes=n_classes, activation='softmax')

# from focal_loss import SparseCategoricalFocalLoss

# Compiling model
# model.compile('Adam', loss = sm.losses.CategoricalCELoss(gamma = 5), metrics = metrics)
# model.compile('Adam', loss = sm.losses.CategoricalFocalLoss(gamma = 3), metrics = metrics)
model.compile('Adam', loss = sm.losses.categorical_crossentropy, metrics = metrics)

print(model.summary())

historyFPNR50 = model.fit(train_img_gen, steps_per_epoch=steps_per_epoch, epochs=50, verbose=1, 
                    validation_data=val_img_gen, validation_steps=val_steps_per_epoch)

model.save('Historyb12FPNR50e50Iou50.hdf5') # Save the model

loss = historyFPNR50.history['loss']
val_loss = historyFPNR50.history['val_loss']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'y', label='Validation Loss')
plt.title('Training Vs Validation Loss scores of FPN R50')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.savefig('LossCurveb12e50FPNR50.png')
plt.show()

iou = historyFPNR50.history['iou_score']
val_iou = historyFPNR50.history['val_iou_score']

plt.plot(epochs, iou, 'r', label='Training Iou')
plt.plot(epochs, val_iou, 'y', label='Validation Iou')
plt.title('Training Vs Validation Iou scores of FPN R50')
plt.xlabel('Epochs')
plt.ylabel('Iou')
plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.savefig('IoUCurveb12e50FPNR50.png')
plt.show()

f1 = historyFPNR50.history['f1-score']
val_f1 = historyFPNR50.history['val_f1-score']

plt.plot(epochs, f1, 'r', label='Training f1-score')
plt.plot(epochs, val_f1, 'y', label='Validation f1-score')
plt.title('Training Vs Validation f1 scores of FPN R50')
plt.xlabel('Epochs')
plt.ylabel('f1-score')
plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.savefig('f1Curveb12e50FPNR50.png')
plt.show()

model = load_model('Historyb12FPNR34e50Iou50.hdf5', compile = False) # Change

k = []
values = []
veg = []
maskk = []
# vegeta = []
# y_true = np.ones(val_steps_per_epoch)
for i in range(int(val_steps_per_epoch)):
    # print(i)
    test_image_batch, test_mask_batch = val_img_gen.__next__()
    
    #Convert categorical to integer for visualization and IoU calculation
    test_mask_batch_argmax = np.argmax(test_mask_batch, axis = 3)
    
    test_pred_batch = model.predict(test_image_batch)
    test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)
    # test_pred_batch_argmax = np.argmax(test_pred_batch)
    
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    # This code could be used to check the %age accuracy in performing prediction for all 
    # classes using confusion matrix
    
    n_classes = 12
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
    m = IOU_keras.result().numpy()
    k.append(m)
    # print("Mean IoU =", m)
        #To calculate I0U for each class...
        
    # m = np.array(k)
    # print("Mean IoU =", m.mean())    
    
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)

    veg.append(np.sum(values[1:,:])/np.sum(values))
    maskk.append(np.sum(values[:,1:])/np.sum(values))    

    # veg.append(np.sum(values[i][1:,:])/np.sum(values[i]))
    # maskk.append(np.sum(values[i][:,1:])/np.sum(values[i]))
    # maskk.append(np.sum(values[:,1:])/np.sum(values[1:,:])*(np.sum(values[1:,:])/np.sum(values)))
    
    # sumD = 0 # vegetation cover initialization
    # # sumB = 0 
    # sumT = np.sum(values) # Total ROI
    # # y_true[i] = (np.sum(values)-np.count_nonzero(test_mask_batch_argmax))/np.sum(values) # Vegetation cover besides frame
    # for i in range(n_classes):

    #     sumT -= values[0,i]
    #     # sumB += values[i,0]
    #     if i > 0:
    #         sumD = sumD + values[i,i]
           
    # veg.append(sumD/sumT)
    
    # maskk.append(np.sum(values[1:,:])/np.sum(values))    
    
    # # maskk.append(np.count_nonzero(test_mask_batch_argmax)/np.sum(values)) # Actual vegetation cover
    # vegeta.append(np.sum(values[:,1:])/np.sum(values)) # predicted vegetation cover
    
    
    # print('Vegetation cover detected: ', sumD/sumT*100)


import pandas as pd
dfFPNR34 = pd.DataFrame({'y_true': veg, 'y_pred': maskk})
dfFPNR34.to_excel('dfFPNR34.xlsx')

import pandas as pd
dfPSPR50 = pd.DataFrame({'y_true': veg, 'y_pred': maskk})
dfPSPR50.to_excel('dfPSPR50.xlsx')

import pandas as pd
dfPSPIRV2 = pd.DataFrame({'y_true': veg, 'y_pred': maskk})
dfPSPIRV2.to_excel('dfPSPIRV2.xlsx') 

import pandas as pd
dfV2u2 = pd.DataFrame({'y_true': veg, 'y_pred': maskk})
dfV2u2.to_excel('dfV2u2.xlsx') 

df34 = pd.DataFrame({'y_true': veg, 'y_pred': maskk})
df34.to_excel('df34.xlsx')
        

dfRx50 = pd.DataFrame({'y_true': veg, 'y_pred': maskk})
dfRx50.to_excel('dfRx50.xlsx')
        
import scipy
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(veg, maskk)

import pandas as pd

df = pd.DataFrame({'y_true': maskk, 'y_pred': veg})

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix

# r2 = r2_score(vegeta, maskk)
# print(r2)
# r2 = r2_score(maskk, vegeta)
# print(r2)

#create basic scatterplot
plt.plot(veg, maskk,  'o')
plt.xlabel('Predicted Vegetation cover')
plt.ylabel('Actual Vegetation cover')

plt.plot(maskk, veg,  'o')
plt.xlabel('Predicted Vegetation cover')
plt.ylabel('Actual Vegetation cover')


#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(veg, maskk, 1)

# vegeta = np.array(vegeta, dtype = 'float64')

#add linear regression line to scatterplot 
# plt.plot(vegeta, m*vegeta+b)



# data = {'y_true': np.round(y_true,2), 'y_pred': np.round(veg,2)}
df2 = pd.DataFrame({'y_true': veg, 'y_pred': maskk})
df2.to_excel('d2.xlsx')
        
# r2 = r2_score(maskk, veg)
r2 = r2_score(veg, maskk)
mse = mean_squared_error(veg, maskk)

print("R-squared:", r2)
print("Mean squared error:", mse)

#create basic scatterplot
plt.plot(veg, maskk,  'o')
plt.xlabel('Predicted Vegetation cover')
plt.ylabel('Actual Vegetation cover')

#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(veg, maskk, 1)

veg = np.array(veg, dtype = 'float64')
#add linear regression line to scatterplot
plt.plot(veg, maskk)
plt.plot(veg, m*veg+b)


r2 = r2_score(maskk, veg)
# r2 = r2_score(veg, maskk)
mse = mean_squared_error(y_true, veg)

print("R-squared:", r2)
print("Mean squared error:", mse)

#create basic scatterplot
plt.plot(maskk, veg,  'o')
plt.ylabel('Predicted Vegetation cover')
plt.xlabel('Actual Vegetation cover')

#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(maskk, veg, 1)
maskk = np.array(maskk, dtype = 'float64')

#add linear regression line to scatterplot 
plt.plot(maskk, m*maskk+b)


model = load_model('Historyb32R34e50.hdf5', compile = False)


# Loading the created DL model
model = load_model('Historyb16RV2e50Iou50.hdf5', compile = False) # Change

test_image_batch, test_mask_batch = val_img_gen.__next__()

#Convert categorical to integer for visualization and IoU calculation
test_mask_batch_argmax = np.argmax(test_mask_batch, axis = 3) 
test_pred_batch = model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)
# test_pred_batch_argmax = np.argmax(test_pred_batch)

# print("--- %s seconds ---" % (time.time() - start_time))

# This code could be used to check the %age accuracy in performing prediction for all 
# classes using confusion matrix

n_classes = 12
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
m = IOU_keras.result().numpy()
print("Mean IoU =", m)

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)

test_image_batch, test_mask_batch = val_img_gen.__next__()

#Convert categorical to integer for visualization and IoU calculation
test_mask_batch_argmax = np.argmax(test_mask_batch, axis = 3)

test_pred_batch = model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)
# test_pred_batch_argmax = np.argmax(test_pred_batch)

# print("--- %s seconds ---" % (time.time() - start_time))

# This code could be used to check the %age accuracy in performing prediction for all 
# classes using confusion matrix

n_classes = 13
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
m = IOU_keras.result().numpy()
# k.append(m)
print("Mean IoU =", m)
# print(values)

# class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0] + values[2,0])
# class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1] + values[2,1])
# class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2] + values[1,2])

# print(class1_IoU) # Background
# print(class2_IoU) # Weeds
# print(class3_IoU) # Soil

# Also generate an image for the predictions performed on test_dataset
img_num = random.randint(0, test_image_batch.shape[0])

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_image_batch[img_num])
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_batch_argmax[img_num], cmap='gray', interpolation='nearest', vmin = 0, vmax = 9)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_pred_batch_argmax[img_num], cmap='gray', interpolation='nearest', vmin = 0, vmax = 9)
plt.show()
