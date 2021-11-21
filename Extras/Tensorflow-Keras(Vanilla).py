# Introduction to Tensorflow framework using Keras - High level API used to train neural networks

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.python.keras.backend import learning_phase, relu

N, D_in, H, D_out = 16, 1000, 100, 10

model = Sequential()
model.add(InputLayer(input_shape=(D_in, )))
model.add(Dense(units = H, activation=relu))
model.add(Dense(units=D_out))

params = model.trainable_variables

loss_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.SGD(learning_rate = 1e-6)

x = tf.random.normal((N, D_in))
y = tf.random.normal((N, D_out))

def step():
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    return loss

for epoch in range(1000):
    opt.minimize(step, params)