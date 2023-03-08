# We'll start with our library imports...
from __future__ import print_function
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
import itertools
from tensorflow.keras.datasets import cifar100

def load_data():
    (train_images, train_lab), (test_images, test_lab) = cifar100.load_data()
    # train_images = train_images.reshape(50000, 3072)
    # test_images = test_images.reshape(10000, 3072)
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_lab = tf.keras.utils.to_categorical(train_lab)
    test_lab = tf.keras.utils.to_categorical(test_lab)

    return train_images, train_lab, test_images, test_lab


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    f = plt.figure()
    #     f.set_figwidth(4)
    #     f.set_figheight(1)
    f.set_size_inches(24, 24)

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
    plt.savefig('test2png.', dpi=100)


def utils_create_accuracy_vs_epochs(history, epochs, title):
    loss_train = history.history['accuracy']
    loss_val = history.history['val_accuracy']
    epochs = range(1, epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training Accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation Accuracy')
    plt.title('Training and Validation Accuracy Architecture - {}'.format(title))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig('1-accuracy_vs_epochs_WithRegularisation_0.01.png')

