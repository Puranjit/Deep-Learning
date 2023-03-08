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
train = tfds.load('mnist', split='train[:90%]', data_dir=DATA_DIR, batch_size=32, shuffle_files=True)

# And the last 10%, we'll hold out as the validation set
# After the training loop, run another loop over this data without the gradient updates to calculate accuracy
validation = tfds.load('mnist', split='train[-10%:]', data_dir=DATA_DIR, batch_size=32, shuffle_files=True)

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()

batch1 = tf.keras.layers.BatchNormalization()
hidden_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_1')
hidden_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_2')
pool_1 = tf.keras.layers.MaxPool2D(padding='same')

batch2 = tf.keras.layers.BatchNormalization()
hidden_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_3')
hidden_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_4')
pool_2 = tf.keras.layers.MaxPool2D(padding='same')
flatten = tf.keras.layers.Flatten()
output = tf.keras.layers.Dense(10)

model = tf.keras.Sequential([batch1, hidden_1, hidden_2, pool_1, batch2, hidden_3, hidden_4, pool_2, flatten, output])

optimizer = tf.keras.optimizers.Adam()

loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 5 epochs
for epoch in range(2):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.cast(batch['image'], tf.float32)
#             print(x)
            labels = batch['label']
            logits = model(x)
#             print(logits)

            # calculate loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
        loss_values.append(loss)
    
        # gradient update
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        # calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        accuracy_values.append(accuracy)

print(model.summary())

# accuracy
print("Accuracy:", np.mean(accuracy_values))


for batch in tqdm(validation):
    predictions = tf.argmax(logits, axis=1)
    val_accu = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
    val_accuracy.append(val_accu)

print("Val Accuracy:", np.mean(val_accuracy))

# Compute confusion matrix
confusion = tf.math.confusion_matrix(labels, predictions)
confusion

# import sklearn
# from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix

# Using Seaborn to plot the Confusion matrix for our results
plt.figure(figsize=(10,8))
import seaborn as sns
sns.heatmap(confusion, annot=True, cmap = 'BuPu')

plt.savefig('Hackathon4.PNG', dpi = 250)
plt.show()