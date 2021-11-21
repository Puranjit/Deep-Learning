import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D

from deep_learning_models import MyCustomModel, functional_model
from my_utils import display_some_image
# from tf.python.keras.layers.convolutional import Conv
# from tf.python.keras.layers.core import Activation

# 1. tensorflow.keras.Sequential
seq_model = tensorflow.keras.Sequential(
    [
        Input(shape=(28,28,1)),
        Conv2D(32, (3,3), activation='relu'),
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3,3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation = 'relu'),
        Dense(10, activation = 'softmax')
    ]
)

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    # print('X_train.shape: ', X_train.shape)
    # print('y_train.shape: ', y_train.shape)
    # print('X_test.shape: ', X_test.shape)
    # print('y_test.shape: ', y_test.shape)
    
    if False:
        display_some_image(X_train, y_train)
    
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255 

    X_train = np.expand_dims(X_train, axis = -1)
    X_test = np.expand_dims(X_test, axis = 3)

    # print('X_train.shape: ', X_train.shape)
    # print('y_train.shape: ', y_train.shape)
    # print('X_test.shape: ', X_test.shape)
    # print('y_test.shape: ', y_test.shape)

    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    # model = functional_model()
    model = MyCustomModel()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics='accuracy')
    
    # Accuracy becomes better and better as we increase the number of hyperparameters to train our model
    # Train, Valid, Test
    
    # model training
    model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

    # Evaluation on test set
    model.evaluate(X_test, y_test, batch_size=64)

