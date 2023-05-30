# -*- coding: utf-8 -*-
"""
Created on Tue May 30 01:04:06 2023

@author: PZS0098
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.layers import AveragePooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def ASPP(inputs):
    
    #  Average pooling
    shape = inputs.shape
    y_pool = AveragePooling2D(pool_size = (shape[1], shape[2]))(inputs)
    # print(shape, y_pool.shape)
    
    #  Convolutional layer
    y_conv = Conv2D(filters = 256, kernel_size = 1, padding = 'same', use_bias = False)(y_pool)
    
    #  Batch Normalization
    y_batch = BatchNormalization()(y_conv)
    
    # ReLU activation
    y_relu = Activation("relu")(y_batch)
    
    # Up-sampling
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation = "bilinear")(y_relu)
    # print(y_pool.shape)
    
    y_1 = Conv2D(filters= 256, kernel_size = 1, padding = "same", use_bias = False)(inputs)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation("relu")(y_1)
    
    y_6 = Conv2D(filters= 256, kernel_size = 1, dilation_rate = 6, padding = "same", use_bias = False)(inputs)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation("relu")(y_6)
    
    y_12 = Conv2D(filters= 256, kernel_size = 1, dilation_rate = 12, padding = "same", use_bias = False)(inputs)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation("relu")(y_12)
    
    y_18 = Conv2D(filters= 256, kernel_size = 1, dilation_rate = 18, padding = "same", use_bias = False)(inputs)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation("relu")(y_18)
    
    # Concatenate all features
    y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])
    # print(y.shape)
    
    y = Conv2D(filters= 256, kernel_size = 1, dilation_rate = 1, padding = "same", use_bias = False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    
    # print(y.shape)  

    return y

def DeepLabV3Plus(shape):
    
    # Inputs
    inputs = Input(shape)
    
    """Pre-trained ResNet"""
    base_model = ResNet50(weights = 'imagenet', include_top = False, input_tensor = inputs)
    
    """Pre-trained ResNet outputs"""
    image_features = base_model.get_layer('conv4_block6_out').output
    # print(image_features.shape)
    
    x_a = ASPP(image_features)
    
    # Up sampling by 4
    x_a = UpSampling2D((4,4), interpolation = "bilinear")(x_a)
    
    """Get low level features"""
    x_b = base_model.get_layer('conv2_block2_out').output
    x_b = Conv2D(filters = 48, kernel_size = 1, padding = 'same', use_bias = False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation("relu")(x_b)
    
    x = Concatenate()([x_a, x_b])
    
    x = Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
        
    x = UpSampling2D((4,4), interpolation = 'bilinear')(x)
    
    # print(x.shape)
    
    # outputs
    
    x = Conv2D(1, (1,1), name  = 'output_layer')(x)
    x = Activation('sigmoid')(x)
    
    """Model"""
    model = Model(inputs = inputs, outputs = x)
    return model
    
    
if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = DeepLabV3Plus(input_shape)
    model.summary()
    