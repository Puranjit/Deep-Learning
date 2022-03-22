# Importing libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# Re-sizing all images
Img_size = [224, 224]

train_path = 'Train\*'
test_path = 'Test'

inception = InceptionV3(input_shape = Img_size + [3], weights = 'imagenet', include_top = False)

for layer in inception.layers:
    layer.trainable = False
    
folders = glob(train_path)

x = Flatten()(inception.output)

prediction = Dense(len(folders), activation = 'softmax')(x)

model = Model(inputs = inception.input, outputs = prediction)

# print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics  = ['accuracy'])

# Data Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Train', target_size = (224,224), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('Test', target_size = (224,224), batch_size = 32, class_mode = 'categorical')

checkpointer = tf.keras.callbacks.ModelCheckpoint('WeedDetectionModelV3.h5', save_best_only = True, verbose = 1)

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = 'logs'),
    tf.keras.callbacks.EarlyStopping(patience = 10, monitor = 'val_loss'),
    checkpointer]

result = model.fit(training_set, validation_data = test_set, epochs = 10, 
                             steps_per_epoch = len(training_set), validation_steps = len(test_set), callbacks = callbacks)

# plot the loss
plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.xticks([x for x in range(0,10)])
plt.title('Loss Vs Validation loss')
plt.savefig('LossVal_lossV3')
plt.show()

# plot the accuracy
plt.plot(result.history['accuracy'], label='train acc')
plt.plot(result.history['val_accuracy'], label='val acc')
plt.xticks([x for x in range(0,10)])
plt.title('Accuracy Vs validation accuracy')
plt.legend()
plt.savefig('AccVal_accV3')
plt.show()

y_pred = model.predict(test_set)
print(y_pred)

y_pred = np.argmax(y_pred, axis=1)
print(y_pred)

model = load_model('WeedDetectionModelV3.h5')