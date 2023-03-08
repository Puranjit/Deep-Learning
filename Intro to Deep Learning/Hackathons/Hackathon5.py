# We'll start with our library imports...
from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops

DATA_DIR = './tensorflow-datasets/'

# The first 90% of the training data
# Loading 90% of the data in the 'mnist' into the training datatset
train = tfds.load('cifar100', split='train[:90%]', data_dir=DATA_DIR, batch_size=64, shuffle_files=True)

# And the last 10%, we'll hold out as the validation set
# After the training loop, run another loop over this data without the gradient updates to calculate accuracy
validation = tfds.load('cifar100', split='train[-10%:]', data_dir=DATA_DIR, batch_size=64, shuffle_files=True)

class ResNetx(tf.Module):
    
    def __init__(self, filter_num, stride = 1):
        super(ResNetx, self).__init__()
        self.stride = stride
        self.filter_num = filter_num

        self.seq = tf.keras.Sequential()

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filter_num*2, kernel_size=1, padding='same', activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filter_num*4, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv3 = tf.keras.layers.Conv2D(filters=self.filter_num*8, kernel_size=1, padding='same', activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(padding='same')

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(filters=self.filter_num*8, kernel_size=1, padding='same', activation=tf.nn.relu)
        self.conv5 = tf.keras.layers.Conv2D(filters=self.filter_num*16, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv6 = tf.keras.layers.Conv2D(filters=self.filter_num*32, kernel_size=1, padding='same', activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(padding='same')

        # self.bn2 = tf.keras.layers.BatchNormalization()
        # self.conv2 = tf.keras.layers.Conv2D(filters=self.filter_num*3, kernel_size=1, padding='same', activation=tf.nn.relu)
        # self.pool2 = tf.keras.layers.MaxPool2D(padding='same')

        self.flatten = tf.keras.layers.Flatten()
        self.output = tf.keras.layers.Dense(100)
        
    def __call__(self, model):
        # using Sequential groups all the layers to run at once

        model = self.seq()
        model = self.bn1(model)
        model = self.conv1(model)
        model = self.conv2(model)
        model = self.conv3(model)
        model = self.pool1(model)

        model = self.bn2(model)
        model = self.conv4(model)
        model = self.conv5(model)
        model = self.conv6(model)
        model = self.pool2(model)

        model = self.flatten(model)
        model = self.output(model)

        # model = tf.keras.Sequential([self.bn1, self.conv1, self.pool1, self.bn2, self.conv2, self.pool2, self.flatten, self.output])

    def model_run(self):

        optimizer = tf.keras.optimizers.Adam()

        loss_values = []
        accuracy_values = []
        val_accuracy = []

        # Loop runs through 5 epochs
        for epoch in range(10):
            for batch in tqdm(train):
                with tf.GradientTape() as tape:
                    # run network
                    x = tf.cast(batch['image'], tf.float32)
        #             print(x)
                    labels = batch['label']
                    logits = self.model(x)
        #             print(logits)

                    # calculate loss
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
                loss_values.append(loss)

                # gradient update
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # calculate accuracy
                predictions = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
                accuracy_values.append(accuracy)

        print(self.model.summary())

        # accuracy
        print("Accuracy:", np.mean(accuracy_values))

        for batch in tqdm(validation):
            predictions = tf.argmax(logits, axis=1)
            val_accu = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
            val_accuracy.append(val_accu)

        print("Val Accuracy:", np.mean(val_accuracy))

mod = ResNetx(16)