from __future__ import print_function

import numpy as np # to use numpy arrays
import tensorflow as tf            #2 to specify and run computation graphs
from tensorflow.keras.datasets import imdb # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from utils_2 import load_data, utils_create_accuracy_vs_epochs
from  model_2 import  model_arch_1, model_arch_2


inputs, targets, input_train = load_data()
val_accuracy, train_accuracy = model_arch_1(inputs, targets, input_train)
utils_create_accuracy_vs_epochs(val_accuracy, train_accuracy)

# Y_pred = model.predict(x_test)
# # Convert predictions classes to one hot vectors
# Y_pred_classes = np.argmax(Y_pred, axis=1)
# # Convert validation observations to one hot vectors
#
# # compute the confusion matrix
# confusion_mtx = confusion_matrix(y_test, Y_pred_classes, normalize='all')
# # plot the confusion matrix
#
#
# # In[36]:
#
#
# sns.heatmap(confusion_mtx, annot=True, fmt='g', cbar=False, cmap="vlag")
# plt.title("confusion matrix for validation")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# # plt.savefig("fmnist/model_best_submitted_to_handin.png", facecolor="white")
# plt.show()
