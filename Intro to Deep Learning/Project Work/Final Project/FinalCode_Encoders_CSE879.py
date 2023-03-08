# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:46:07 2022
Final code for CSE879 : Intro to deep learning
Team : Encoders
@author: psingh24
"""

# Importing libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
from glob import glob

from sklearn import metrics
import seaborn as sns

# Re-sizing all images
Img_size = [224, 224]

train_path = 'Train\*'
test_path = 'Test\*'

# Architecture being used
resnet = ResNet50(input_shape = Img_size + [3], weights = 'imagenet', include_top = False)
# inception = InceptionResNetV2(input_shape = Img_size + [3], weights = 'imagenet', include_top = False)

# Choice : Update/freeze weights during training
    # True - weights in the layers are updated at each iteration
    # False - weights in the layers are not updated at each iteration
for layer in resnet.layers:
    # layer.trainable = False
    layer.trainable = True
    
folders = glob(train_path)

x = Flatten()(resnet.output)

prediction = Dense(len(folders), activation = 'softmax')(x)

model = Model(inputs = resnet.input, outputs = prediction)

# print(model.summary())

# Defining hyper-parameters for the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics  = ['accuracy'])

# Data Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Reading images
training_set = train_datagen.flow_from_directory('Train', target_size = (224,224), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('Test', target_size = (224,224), batch_size = 32, class_mode = 'categorical')

# Defining checkpointers to save the best model
checkpointer = tf.keras.callbacks.ModelCheckpoint('WeedDetectionModelV3.h5', save_best_only = True, verbose = 1)

# Defining callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = 'logs'),
    tf.keras.callbacks.EarlyStopping(patience = 5, monitor = 'val_loss'),
    checkpointer]

# Training the model
result = model.fit(training_set, validation_data = test_set, epochs = 25, steps_per_epoch = len(training_set), validation_steps = len(test_set), callbacks = callbacks)

# model.save('WeedDetectionModelIncResV2.h5')
model.save('WeedDetectionModelR50.h5')

# plot the loss
plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Categorical crossentropy Error')
plt.xticks([x for x in range(0,8)])
plt.title('Training Vs Validation loss')
plt.savefig('LossVal_lossFineTuneV3')
plt.show()

# plot the accuracy
plt.plot(result.history['accuracy'], label='train acc')
plt.plot(result.history['val_accuracy'], label='val acc')
plt.xticks([x for x in range(0,8)])
plt.xlabel('Epochs')
plt.ylabel('Accuracy values')
plt.title('Training Vs validation accuracy')
plt.legend()
plt.savefig('AccVal_accFineTuneV3')
plt.show()

# Running prediction on test dataset
y_pred = model.predict(test_set)
print(y_pred)

# Converting results from predicition
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)

# Method to calculate the confusion matrix
testing = []
for i in range(len(test_set)):
    testing.append(list(np.argmax(test_set[i][1], axis=1)))

actual = []
for i in range(len(testing)):
    for j in testing[i]:
        actual.append(j)
actual = np.array(actual)

# Printing the confusion matrix
confusion_matrix = metrics.confusion_matrix(actual, y_pred)
confusion_matrix = confusion_matrix/1527

# Printing classification results
print(metrics.classification_report(actual, y_pred))

# Plotting the confusion matrix
sns.heatmap(confusion_matrix, annot = True)
