from __future__ import print_function

import numpy as np # to use numpy arrays
import tensorflow as tf            #2 to specify and run computation graphs
from tensorflow.keras.datasets import imdb # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold


def load_data():
    (input_train, target_train), (input_test, target_test) = imdb.load_data(num_words=20000)

    input_train = pad_sequences(input_train, maxlen=100)
    input_test = pad_sequences(input_test, maxlen=100)

    inputs = np.concatenate((input_train, input_test), axis=0)
    targets = np.concatenate((target_train, target_test), axis=0)

    return inputs, targets, input_train


def utils_create_accuracy_vs_epochs(val_accuracy, train_accuracy):
    val_acc = np.array(val_accuracy)

    train_acc = np.array(train_accuracy)

    mean_train_acc = list(train_acc.mean(0))

    mean_val_acc = list(val_acc.mean(0))

    epochs = range(1, epochs + 1)
    plt.plot(epochs, mean_train_acc, 'g', label='Training Accuracy')
    plt.plot(epochs, mean_val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Architecture ')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig('Accuracy_vs_epochs')


def utils_create_accuracy_vs_epochs(history, epochs, title):
    loss_train = history.history['accuracy']
    loss_val = history.history['val_accuracy']
    epochs = range(1, epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training Accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation Accuracy')
    plt.title('Training and Validation Accuracy Architecture - {}'.format(title))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('accuracy_vs_epochs.png')



# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True')
#     plt.xlabel('Predicted')

