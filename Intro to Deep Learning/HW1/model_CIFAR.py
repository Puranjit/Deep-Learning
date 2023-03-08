#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def first_architecture_eff_net_b6(initializer = 'orthogonal', activation_func = 'elu', neurons_in_hidden = 256, 
                      batch_normalization = False, dropout = False, dropout_proportion = 0.5, number_hidden_layers = 1):
        import tensorflow as tf
        import numpy as np
        tf.keras.backend.clear_session()
        np.random.seed(84)
        tf.random.set_seed(84)
    
        #early stopping to monitor the validation loss and avoid overfitting
        # early_stop = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=5, restore_best_weights=True)
    
        efnb0 = tf.keras.applications.efficientnet.EfficientNetB6(weights='imagenet', include_top=False, input_shape=(528,528,3), classes=100)
        # efnb0.trainable = False
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(32, 32, 3)))
        model.add(tf.keras.layers.Lambda(lambda x: tf.keras.applications.efficientnet.preprocess_input(x)))
        model.add(tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (528,528))))
        
        model.add(efnb0)
        
        model.add(tf.keras.layers.Conv2D(3000,3, padding = "same",kernel_regularizer=tf.keras.regularizers.L2()))
        model.add(tf.keras.layers.MaxPool2D(2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        
        model.add(tf.keras.layers.Conv2D(4000,3, padding = "same",kernel_regularizer=tf.keras.regularizers.L2()))
        model.add(tf.keras.layers.MaxPool2D(2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        
        model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2()))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        
          # model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2()))
          # model.add(tf.keras.layers.BatchNormalization())
          # model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(100, activation='softmax', kernel_regularizer=tf.keras.regularizers.L2()))
        
        
          #model compiling
        return(model)


def second_architecture_vgg_19(initializer = 'orthogonal', activation_func = 'elu', neurons_in_hidden = 256, 
                      batch_normalization = False, dropout = False, dropout_proportion = 0.5, number_hidden_layers = 1):
        import tensorflow as tf
        import numpy as np
        tf.keras.backend.clear_session()
        np.random.seed(84)
        tf.random.set_seed(84)
    
        #early stopping to monitor the validation loss and avoid overfitting
        # early_stop = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=5, restore_best_weights=True)
    
        efnb0 = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3), classes=100)
        # efnb0.trainable = False
        
        efnb0.trainable = False
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(32, 32, 3)))
        model.add(tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg19.preprocess_input(x)))
        model.add(tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (224,224))))
        model.add(efnb0)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2()))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2()))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(100, activation='softmax', kernel_regularizer=tf.keras.regularizers.L2()))
        
        
          #model compiling
        return(model)
