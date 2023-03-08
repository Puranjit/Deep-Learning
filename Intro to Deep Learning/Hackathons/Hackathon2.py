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

print('Example 1')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(200, tf.nn.relu))
model.add(tf.keras.layers.Dense(10))

optimizer = tf.keras.optimizers.Adam()

loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 5 epochs
for epoch in range(5):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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

print('Example 2')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, tf.nn.relu))
model.add(tf.keras.layers.Dense(10))
optimizer = tf.keras.optimizers.Adam()


loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 5 epochs
for epoch in range(5):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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

print('Example 3')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(200, tf.nn.tanh))
model.add(tf.keras.layers.Dense(10))

optimizer = tf.keras.optimizers.Adagrad()

loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 5 epochs
for epoch in range(5):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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

print('Example 4')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, tf.nn.tanh))
model.add(tf.keras.layers.Dense(10))
optimizer = tf.keras.optimizers.Adagrad()


loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 5 epochs
for epoch in range(5):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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


print('Example 5')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(200, tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(10))

optimizer = tf.keras.optimizers.SGD()

loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 5 epochs
for epoch in range(5):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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

print('Example 6')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(10))
optimizer = tf.keras.optimizers.SGD()


loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 5 epochs
for epoch in range(5):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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

print('Example 7')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(200, tf.nn.relu))
model.add(tf.keras.layers.Dense(50, tf.nn.relu))
model.add(tf.keras.layers.Dense(10))

optimizer = tf.keras.optimizers.Adam()

loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 5 epochs
for epoch in range(5):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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

print('Example 8')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(25, tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(10))
optimizer = tf.keras.optimizers.Adam()


loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 5 epochs
for epoch in range(5):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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

print('Example 9')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(200, tf.nn.relu))
model.add(tf.keras.layers.Dense(50, tf.nn.relu))
model.add(tf.keras.layers.Dense(10))

optimizer = tf.keras.optimizers.RMSprop()

loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 5 epochs
for epoch in range(5):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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

print('Example 10')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(25, tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(10))
optimizer = tf.keras.optimizers.RMSprop()


loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 5 epochs
for epoch in range(5):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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

print('Example 11')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(200, tf.nn.relu))
model.add(tf.keras.layers.Dense(10))
optimizer = tf.keras.optimizers.Adam()


loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 10 epochs
for epoch in range(10):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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

print('Example 12')

# using Sequential groups all the layers to run at once
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(200, tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(10))
optimizer = tf.keras.optimizers.Adam()


loss_values = []
accuracy_values = []
val_accuracy = []

# Loop runs through 10 epochs
for epoch in range(10):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

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