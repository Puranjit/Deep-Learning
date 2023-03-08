# We'll start with our library imports...
from __future__ import print_function

import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

def load_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    train_lab = tf.keras.utils.to_categorical(train_labels)
    test_images = test_images / 255.0

    test_labels = tf.keras.utils.to_categorical(test_labels)

    return train_images, train_lab, test_images, test_labels


def utils_create_loss_vs_epochs(history, epochs, title):
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1, epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss Architecture - {}'.format(title))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('loss_vs_epochs.png')


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


def utils_plot_different_learning_rates(history, epochs, title):
    k = []
    print('title', title)
    epochs = range(1, epochs + 1)
    loss_val_0 = history[0].history['val_accuracy']
    a, = plt.plot(epochs, loss_val_0)
    loss_val_1 = history[1].history['val_accuracy']
    b, = plt.plot(epochs, loss_val_1)
    loss_val_2 = history[2].history['val_accuracy']
    c, = plt.plot(epochs, loss_val_2)

    plt.title('Validation Accuracy VS Learning Rate'.format(title))

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #     plt.legend(k,title)
    plt.legend([a, b, c], title)

    plt.show()

    plt.savefig('learningRate_vs_Accuracy.png')



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

