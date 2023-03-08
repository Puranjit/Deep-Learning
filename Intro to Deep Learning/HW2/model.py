from __future__ import print_function

import numpy as np # to use numpy arrays
import tensorflow as tf            #2 to specify and run computation graphs
from tensorflow.keras.datasets import imdb # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold




def model_arch_1(inputs, targets, input_train):

    vocab_size = 20000
    embed_size = 128
    num_folds = 2
    all_history = []
    acc_per_fold = []
    loss_per_fold = []
    kfold = KFold(n_splits=num_folds, shuffle=True)
    query_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        padding='same')
    #  kernel_regularizer=tf.keras.regularizers.L2(0.001)
    value_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        padding='same')
    # kernel_regularizer=tf.keras.regularizers.L2(0.001)
    attention = tf.keras.layers.Attention()
    concat = tf.keras.layers.Concatenate()
    cells = [tf.keras.layers.LSTMCell(28), tf.keras.layers.LSTMCell(14)]
    rnn = tf.keras.layers.RNN(cells)
    output_layer = tf.keras.layers.Dense(1)

    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        inp = tf.keras.layers.Input(shape=(None,))
        embd = tf.keras.layers.Embedding(vocab_size, embed_size, input_shape=(input_train.shape[1],))(inp)
        query = query_layer(embd)
        value = value_layer(embd)
        query_value_attention = attention([query, value])
        attended_values = concat([query, query_value_attention])
        RNN_cell = output_layer(rnn(attended_values))
        # dense = tf.keras.layers.Dense(units=1, name='Dense')(RNN_cell)
        model = tf.keras.models.Model(inputs=inp, outputs=[RNN_cell])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        history = model.fit(inputs[train], targets[train], epochs=6, batch_size=256, verbose=2,
                            validation_data=(inputs[test], targets[test]))
        all_history.append(history)
        #     scores = model.evaluate(inputs[test], targets[test], verbose=0)
        #     print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        #     acc_per_fold.append(scores[1])
        #     loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')

    # In[13]:

    # cells = [tf.keras.layers.LSTMCell(256), tf.keras.layers.LSTMCell(64)]
    # query_layer = tf.keras.layers.Conv1D(
    #     filters=100,
    #     kernel_size=4,
    #     padding='same')
    # value_layer = tf.keras.layers.Conv1D(
    #     filters=100,
    #     kernel_size=4,
    #     padding='same')
    # model = Sequential()
    # model.add(Embedding(vocab_size, embed_size, input_shape = (x_train.shape[1],)))
    # model.add(RNN(cells))
    # # model.add(query_layer)
    # # model.add(value_layer)
    # model.add(MultiHeadAttention

    # model.add(Dense(1))
    epochs = 6
    # print(history.history)
    val_accuracy = []
    train_accuracy = []
    for x in all_history:
        val_accuracy.append(x.history['val_accuracy'])
        train_accuracy.append(x.history['accuracy'])


    return val_accuracy, train_accuracy


def model_arch_2(inputs, targets, input_train):
    num_folds = 2
    all_history = []
    acc_per_fold = []
    loss_per_fold = []
    kfold = KFold(n_splits=num_folds, shuffle=True)
    query_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        padding='same', kernel_regularizer=tf.keras.regularizers.L2())
    value_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        padding='same', kernel_regularizer=tf.keras.regularizers.L2())
    attention = tf.keras.layers.Attention()
    concat = tf.keras.layers.Concatenate()
    cells = [tf.keras.layers.LSTMCell(128), tf.keras.layers.LSTMCell(64)]
    rnn = tf.keras.layers.RNN(cells)
    output_layer = tf.keras.layers.Dense(128)
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        inp = tf.keras.layers.Input(shape=(None,))
        embd = tf.keras.layers.Embedding(vocab_size, embed_size, input_shape=(input_train.shape[1],))(inp)
        conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='relu')(embd)
        MP1 = tf.keras.layers.MaxPool1D(2)(conv1)
        conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=4, padding='same', activation='relu')(MP1)
        MP2 = tf.keras.layers.MaxPool1D(2)(conv2)
        query = query_layer(MP2)
        value = value_layer(MP2)
        query_value_attention = attention([query, value])
        attended_values = concat([query, query_value_attention])
        RNN_cell = output_layer(rnn(attended_values))
        drop1 = tf.keras.layers.Dropout(0.3)(RNN_cell)
        dense_1 = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.L2())(drop1)
        drop2 = tf.keras.layers.Dropout(0.3)(dense_1)
        dense = tf.keras.layers.Dense(units=1, name='Dense')(drop2)
        model = tf.keras.models.Model(inputs=inp, outputs=[dense])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        history = model.fit(inputs[train], targets[train], epochs=10, batch_size=32, verbose=2)
        all_history.append(history)
        scores = model.evaluate(inputs[test], targets[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        fold_no = fold_no + 1
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')