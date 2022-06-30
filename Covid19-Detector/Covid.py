import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from IPython.display import Image
from PIL import Image
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from keras_tuner.tuners import RandomSearch

train_image_generator = ImageDataGenerator(rescale = 1.0/255)
test_image_generator = ImageDataGenerator(rescale = 1.0/255)

training_images = train_image_generator.flow_from_directory(
                                                           'Covid19Dataset/train',
                                                           target_size = (64,64),
                                                           batch_size = 8,
                                                           class_mode = 'binary')
testing_images = test_image_generator.flow_from_directory('Covid19Dataset/test',
                                                           target_size = (64,64),
                                                           batch_size = 8,
                                                           class_mode = 'binary')

def plotImages(images):
    fig, axes = plt.subplots(1,5,figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

sample_training_images, _ = next(training_images)
plotImages(sample_training_images[:5])

# Creation of CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Further Adding the NeuralNet in ConvNet Model
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# TRAINING OF A MODEL
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit_generator(training_images, epochs = 10, validation_data = testing_images)

# Visualizing loss and accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc='lower right')
plt.title('ACCURACY')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc='upper right')
plt.title('LOSS')

plt.show()

# Saving the model
model.save('model.h5')

from tensorflow.keras.models import load_model
import cv2
import numpy as np

# We need not to create and train the model again
# No need to train the model, its a pre defined trained model
model = load_model("model.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Testing the Normal Image
image = cv2.imread("Covid19Dataset/test/normal/NORMAL2-IM-1385-0001.jpeg")    # 1
# image = cv2.imread("covid19dataset/test/covid/nejmoa2001191_f3-PA.jpeg")    # 0
image = cv2.resize(image, (64, 64))
image = np.reshape(image, [1, 64, 64, 3])

# classes = model.predict_classes(image)
classes = (model.predict(image) > 0.5).astype("int32")
label = ["COVID-19 INFECTED", "NORMAL"]
print(classes)
print(label[classes[0][0]])

"""
    Putting Up Image Classification from the Saved Model
    Alongwith Flask App
"""

from flask import *

from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Creating a Python App running on Flask Server
app = Flask(__name__)

def predictCOVID(imageToBeTested):

    model = load_model("model.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    image = cv2.imread(imageToBeTested)
    image = cv2.resize(image, (64, 64))
    image = np.reshape(image, [1, 64, 64, 3])

    classes = (model.predict(image) > 0.5).astype("int32")
    # classes = model.predict_classes(image)  # [[0]]

    label = ["COVID-19 INFECTED", "NORMAL"]

    return label[classes[0][0]]


@app.route('/')
def index():
    return render_template("image-classification-index.html")

@app.route('/upload-image', methods=['POST'])
def uploadImage():
    if request.method == 'POST': # Just to Validate if user is uploading the file in POST Request
        file = request.files['image']
        file.save(file.filename)

        label = predictCOVID(file.filename)

        return render_template('image-classification-result.html', name=label)


if __name__ == '__main__':
    # app.run() # execute the app i.e. let the app run on Flask Server
    app.run(debug=True)     # Enable Debugging for the error 