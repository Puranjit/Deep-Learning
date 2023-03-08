# Accuracy vs Epochs

import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
from model_CIFAR import model_architecture_1, ResNet34
from util_CIFAR import load_data, plot_confusion_matrix, utils_create_accuracy_vs_epochs


train_images, train_lab, test_images, test_lab = load_data()
model = ResNet34()
# model = model_architecture_1()


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy',  metrics= ['accuracy'])
epochs = 50

history1 = model.fit(train_images, train_lab, epochs=epochs, validation_data=(test_images, test_lab), batch_size=32)

utils_create_accuracy_vs_epochs(history=history1, epochs=epochs, title=' ')


Y_pred = model.predict(test_images)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(test_lab, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx,
                      classes=[x for x in range(1, 101)])

