# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 01:55:46 2022

@author: psingh24
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
train_img_path = "Data_PSP/" # Change
train_mask_path = "Data_PSP/" # Change


img_list = os.listdir(train_img_path)
msk_list = os.listdir(train_mask_path)

num_images = len(os.listdir(train_img_path))

# img_num = random.randint(0, num_images-1)

# img_for_plot = cv2.imread(train_img_path+img_list[img_num], 1)
# img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)

# mask_for_plot = cv2.imread(train_mask_path+msk_list[img_num], 0)

# plt.figure(figsize=(12, 8))
# plt.subplot(121)
# plt.imshow(img_for_plot)
# plt.title('Image')
# plt.subplot(122)
# plt.imshow(mask_for_plot, cmap='gray')
# plt.title('Mask')
# plt.show()

# Defining batch_size and total number of pixel associated classes in our annotated dataset
seed = 24
batch_size = 32
n_classes = 12

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from tensorflow.keras.utils import to_categorical

# Defining backbone

# Use this to preprocess input for transfer learning
BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)

#Define a function to perform additional preprocessing after datagen.
#For example, scale images, convert masks to categorical, etc. 
def preprocess_data(img, mask, n_classes):
    #Scale images
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    # img = preprocess_input(img)  #Preprocess based on the pretrained backbone...
    #Convert mask to one-hot
    # print(np.unique(mask))
    mask = to_categorical(mask, n_classes)
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
        target_size = (240,240),
        # color_mode='rgb',
        interpolation='nearest',
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        target_size = (240,240),
        interpolation='nearest',
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
       
    for (img, mask) in train_generator:
    
        k = list(np.unique(mask))
        
        for i in range(len(k)):
            count = np.count_nonzero(mask == k[i])
            if count < 5000:
                mask = np.where(mask == k[i], 0, mask)
                
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)

# Run the code to perform data augmentation on train dataset
train_img_path = "Data_PSP/" # Change
train_mask_path = "Data_PSP/" # Change
train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class = 12)

# Run the code to perform data augmentation on train dataset
val_img_path = "Data_PSP/" # Change
val_mask_path = "Data_PSP/" # Change
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class = 12)

x, y = train_img_gen.__next__()

for i in range(0,3):
    image = x[i]
    mask = np.argmax(y[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()

x_val, y_val = val_img_gen.__next__()

for i in range(0,3):
    image = x_val[i]
    mask = np.argmax(y_val[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()


num_train_imgs = len(os.listdir('Data_PSP/train_images/train/')) # Change
num_val_images = len(os.listdir('Data_PSP/val_images/val/')) # Change

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

model = sm.PSPNet(BACKBONE, encoder_weights='imagenet',
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                classes=n_classes, activation='softmax')

# from focal_loss import SparseCategoricalFocalLoss
# Compiling model
model.compile('Adam', loss = sm.losses.categorical_crossentropy, metrics = metrics)

print(model.summary())

historyPsp = model.fit(train_img_gen, steps_per_epoch=steps_per_epoch, epochs=50, verbose=1, 
                    validation_data=val_img_gen, validation_steps=val_steps_per_epoch)

model.save('Historyb32PSPR50e50Iou50.hdf5') # Save the model

loss = historyPsp.history['loss']
val_loss = historyPsp.history['val_loss']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'y', label='Validation Loss')
plt.title('Training Vs Validation Loss scores of Rx50')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.savefig('LossCurveb16e50Rx50.png')
plt.show()

iou = historyPsp.history['iou_score']
val_iou = historyPsp.history['val_iou_score']

plt.plot(epochs, iou, 'r', label='Training Iou')
plt.plot(epochs, val_iou, 'y', label='Validation Iou')
plt.title('Training Vs Validation Iou scores of IV3')
plt.xlabel('Epochs')
plt.ylabel('Iou')
plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.savefig('IoUCurveb16e50Iv3.png')
plt.show()

f1 = historyPsp.history['f1-score']
val_f1 = historyPsp.history['val_f1-score']

plt.plot(epochs, f1, 'r', label='Training f1-score')
plt.plot(epochs, val_f1, 'y', label='Validation f1-score')
plt.title('Training Vs Validation f1 scores of Rx50')
plt.xlabel('Epochs')
plt.ylabel('f1-score')
plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.savefig('f1Curveb16e50Rx50.png')
plt.show()

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

n_classes = 12 # Change
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
m = IOU_keras.result().numpy()
print("Mean IoU =", m)

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)

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
