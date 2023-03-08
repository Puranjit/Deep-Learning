from __future__ import print_function

import numpy as np  # to use numpy arrays
import itertools
import tensorflow as tf  # to specify and run computation graphs
# import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from model_FMNIST import create_model_1, create_model_2_lr1, create_model_2_lr2, create_model_2
from util_FMNIST import utils_create_accuracy_vs_epochs, utils_create_loss_vs_epochs, utils_plot_different_learning_rates, plot_confusion_matrix, load_data

train_images, train_labels, test_images, test_labels = load_data()
optimizer1, model1 = create_model_1()
optimizer2, model2 = create_model_2()
optimizer_lr1, model_lr1 = create_model_2_lr1()
optimizer_lr2, model_lr2 = create_model_2_lr2()


epochs = 100

model1.compile(optimizer=optimizer1,
               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
history1 = model1.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
utils_create_loss_vs_epochs(history1, epochs=epochs, title='1')
utils_create_accuracy_vs_epochs(history1, epochs=epochs, title='1')

model2.compile(optimizer=optimizer2,
               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
history2 = model2.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

utils_create_loss_vs_epochs(history2, epochs=epochs, title='2')
# print('\nTest accuracy:', test_acc)
utils_create_accuracy_vs_epochs(history2, epochs=epochs, title='2')
test_loss, test_acc = model2.evaluate(test_images, test_labels, verbose=2)

model_lr1.compile(optimizer=optimizer_lr1,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
history_lr1 = model_lr1.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

model_lr2.compile(optimizer=optimizer_lr2,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
history_lr2 = model_lr2.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

history_diff_lr = [history_lr1, history2, history_lr2]
titles = ['LR: 0.01', 'LR: 0.001', 'LR: 0.0001']
utils_plot_different_learning_rates(history_diff_lr, epochs, titles)
print('\nTest accuracy:', test_acc)



Y_pred = model2.predict(test_images)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(test_labels, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx,
                      classes=['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
                               'Bag', 'Ankle Boot'])
