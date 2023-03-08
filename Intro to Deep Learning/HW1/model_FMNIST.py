import tensorflow as tf


def create_model_1():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(200, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                    activity_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # model = tf.keras.Sequential([
    #
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(10)
    # ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    return optimizer, model


def create_model_2_without_reg():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # model = tf.keras.Sequential([
    #
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(10)
    # ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    return optimizer, model


def create_model_2():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(200, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-2, l2=1e-2)))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.10))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # model = tf.keras.Sequential([
    #
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(10)
    # ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    return optimizer, model


def create_model_2_lr1():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(200, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-2, l2=1e-2)))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.10))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # model = tf.keras.Sequential([
    #
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(10)
    # ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    return optimizer, model


def create_model_2_lr2():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(200, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-2, l2=1e-2)))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.10))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # model = tf.keras.Sequential([
    #
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(10)
    # ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    return optimizer, model
