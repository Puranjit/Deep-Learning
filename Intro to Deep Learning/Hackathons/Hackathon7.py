# I have used different code sizes to calculate Mean squared error after running the autoencoders

# We'll start with our library imports...
from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops

from tensorflow.keras.losses import MeanSquaredError

DATA_DIR = './tensorflow-datasets/'

# Let's use the code from Hack2 to load MNIST
ds = tfds.load('mnist', shuffle_files=True, data_dir = DATA_DIR) # this loads a dict with the datasets

# We can create an iterator from each dataset
# This one iterates through the train data, shuffling and minibatching by 32
train_ds = ds['train'].shuffle(1024).batch(32)

def upscale_block(filters, kernel_size=3, scale=2, activation=tf.nn.elu):
    """[Sub-Pixel Convolution](https://arxiv.org/abs/1609.05158)"""
    # Increase the number of channels to the number of channels times the scale squared
    conv = tf.keras.layers.Conv2D(filters * (scale**2),
                                  (kernel_size, kernel_size),
                                  activation=activation,
                                  padding='same')
    # Rearrange blocks of (1,1,scale**2) pixels into (scale,scale,1) pixels
    rearrange = tf.keras.layers.Lambda(
        lambda x: tf.nn.depth_to_space(x, scale))
    return tf.keras.Sequential([conv, rearrange])


class UpscaleBlock(tf.keras.layers.Layer):
    def __init__(self, number, kernel_size=3, activation=tf.nn.swish):
        super().__init__(name="UpscaleBlock" + str(number))
        self.activation = activation
        self.kernel_size = kernel_size
        self.is_built = False

    def build(self, x):
        channels = x.shape.as_list()[-1]
        filters = channels // 2

        bn1 = tf.keras.layers.BatchNormalization()
        conv1 = upscale_block(filters)
        bn2 = tf.keras.layers.BatchNormalization()
        conv2 = tf.keras.layers.Conv2D(filters,
                                       self.kernel_size,
                                       padding='same')
        self.main_network = [self.activation, bn1, conv1, self.activation, bn2, conv2]

        self.skip_connection = upscale_block(filters)
        self.se_activate = SqueezeExcite(filters)
        self.is_built = True

    def __call__(self, input_):
        if not self.is_built:
            self.build(input_)
        x = input_
        for layer in self.main_network:
            x = layer(x)
        output = x
        skip = self.skip_connection(input_)
        return skip + 0.1 * output


class FactorizedReduce(tf.Module):
    """Downscale version of the sub-pixel convolution which re-arranges pixels"""
    def __init__(self, channels):
        super(FactorizedReduce, self).__init__()
        assert channels % 2 == 0
        self.conv_1 = tf.keras.layers.Conv2D(channels // 4, 1, strides=2)
        self.conv_2 = tf.keras.layers.Conv2D(channels // 4, 1, strides=2)
        self.conv_3 = tf.keras.layers.Conv2D(channels // 4, 1, strides=2)
        self.conv_4 = tf.keras.layers.Conv2D(channels - 3 * (channels // 4),
                                             1,
                                             strides=2)
        self.convs = [self.conv_1, self.conv_2, self.conv_3, self.conv_4]

    def __call__(self, x):
        """Assumes NHCW data"""
        assert x.shape[2] > 1
        assert x.shape[3] > 1
        out = tf.nn.swish(x)
        conv1 = self.conv_1(out)
        conv2 = self.conv_2(out[:, :, 1:, 1:])
        conv3 = self.conv_3(out[:, :, :, 1:])
        conv4 = self.conv_4(out[:, :, 1:, :])
        out = tf.concat([conv1, conv2, conv3, conv4], -1)
        return out


class SqueezeExcite(tf.Module):
    """Activation function that performs gating"""
    def __init__(self, out_channels):
        super().__init__()
        num_hidden = max(out_channels // 16, 4)
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hidden), tf.keras.layers.Lambda(tf.nn.relu),
            tf.keras.layers.Dense(out_channels), tf.keras.layers.Lambda(tf.nn.sigmoid)
        ])

    def __call__(self, x):
        """The choice of axes assumes we're working with NHWC data"""
        ax = tf.math.reduce_mean(x, axis=[1, 2])
        # data should be flat at this po,int
        bx = self.net(ax)
        cx = tf.expand_dims(tf.expand_dims(bx, 1), 1)
        return cx * x


class DownscaleBlock(tf.keras.layers.Layer):
    def __init__(self, number, kernel_size=3, activation=tf.nn.swish):
        super().__init__(name="DownscaleBlock" + str(number))
        self.activation = activation
        self.kernel_size = kernel_size
        self.is_built = False

    def build(self, x):
        channels = x.shape.as_list()[-1]
        filters = channels * 2

        bn1 = tf.keras.layers.BatchNormalization()
        conv1 = tf.keras.layers.Conv2D(filters,
                                       self.kernel_size,
                                       strides=2,
                                       padding='same')
        bn2 = tf.keras.layers.BatchNormalization()
        conv2 = tf.keras.layers.Conv2D(filters,
                                       self.kernel_size,
                                       padding='same')
        self.main_network = [self.activation, bn1, conv1, self.activation, bn2, conv2]

        self.skip_connection = FactorizedReduce(filters)
        self.se_activate = SqueezeExcite(filters)
        self.is_built = True

    def __call__(self, input_):
        if not self.is_built:
            self.build(input_)
        x = input_
        for layer in self.main_network:
            x = layer(x)
        output = x
        skip = self.skip_connection(input_)
        return skip + 0.1 * output


encoder_network = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same',
                           activation=tf.nn.swish),  #28,28,16
    DownscaleBlock(1),  # 14,14,32
    # DownscaleBlock(2),  # 7,7,64
    tf.keras.layers.Conv2D(64, 3, padding='same',
                           activation=tf.nn.swish),  # 7,7,64
    tf.keras.layers.Conv2D(16, 3, padding='same',
                           activation=tf.nn.swish),  # 7,7,16
    tf.keras.layers.Conv2D(1, 3, padding='same'),  # 7,7,4
])

decoder_network = tf.keras.Sequential([
    tf.keras.layers.Conv2D(4, 3, padding='same',
                           activation=tf.nn.swish),  # 7,7,4
    tf.keras.layers.Conv2D(16, 3, padding='same',
                           activation=tf.nn.swish),  # 7,7,16
    tf.keras.layers.Conv2D(64, 3, padding='same',
                           activation=tf.nn.swish),  # 7,7,64
    UpscaleBlock(1),  # 14,14,32
    # UpscaleBlock(2),  # 28,28,16
    tf.keras.layers.Conv2D(4, 3, padding='same',
                           activation=tf.nn.swish),  #28,28,4
    tf.keras.layers.Conv2D(1, 3, padding='same'),  #28,28,16
])

def sparse_autoencoder_loss(x, code, x_hat, sparsity_coeff=5.):
    sparsity_loss = tf.norm(code, ord=1, axis=1)
    reconstruction_loss = tf.reduce_mean(tf.square(x_hat - x)) # Mean Square Error
    total_loss = reconstruction_loss + sparsity_coeff * sparsity_loss
    return total_loss


optimizer = tf.keras.optimizers.Adam()
for batch in tqdm(train_ds):
    with tf.GradientTape() as tape:
        x = tf.cast(batch['image'], tf.float32)
        code = encoder_network(x)
        output = decoder_network(code)
        loss = sparse_autoencoder_loss(x, code, output)
    gradient = tape.gradient(loss, encoder_network.trainable_variables + decoder_network.trainable_variables)
    optimizer.apply_gradients(zip(gradient, encoder_network.trainable_variables + decoder_network.trainable_variables))

mse = MeanSquaredError()
ans = mse(x, output).numpy()

print(ans)

x = np.array([7, 14, 28])

y = np.array([291.135, 103.444534, 27.883097])

plt.plot(x, y, 'o')
plt.xticks([x for x in range(7,29,7) if x%21!=0])

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x+b)

plt.ylabel('Mean Square Error')
plt.xlabel('Different Code sizes')
plt.title('Best fit line for autoencoder architecture')

plt.show()