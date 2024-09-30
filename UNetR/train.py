import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from patchify import patchify
from unetr_2d import build_unetr_2d
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K

""" UNETR  Configration """
cf = {}
cf["image_size"] = 256
cf["num_channels"] = 3
cf["patch_size"] = 256

cf["num_classes"] = 8

cf["num_layers"] = 16
cf["hidden_dim"] = 256
cf["mlp_dim"] = 512
cf["num_heads"] = 8
cf["dropout_rate"] = 0.15

cf["num_patches"] = (cf["image_size"]**2)//(cf["patch_size"]**2)
cf["flat_patches_shape"] = (
    cf["num_patches"],
    cf["patch_size"]*cf["patch_size"]*cf["num_channels"]
)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):
    train_x = sorted(glob(os.path.join(path, "train", "images", "*.png")))
    train_y = sorted(glob(os.path.join(path, "train", "labels", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "val", "images", "*.png")))
    valid_y = sorted(glob(os.path.join(path, "val", "labels", "*.png")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.png")))
    test_y = sorted(glob(os.path.join(path, "test", "labels", "*.png")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    image = image / 255.0

    """ Processing to patches """
    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(image, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"])
    patches = patches.astype(np.float32)

    return patches

def read_mask(path):
    path = path.decode()
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]))
    mask = mask.astype(np.int32)
    return mask

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        y = tf.one_hot(y, cf["num_classes"])
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape(cf["flat_patches_shape"])
    y.set_shape([cf["image_size"], cf["image_size"], cf["num_classes"]])
    return x, y

# Custom IoU metric
def iou_score(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred - y_true * y_pred)
    return intersection / (union + tf.keras.backend.epsilon())

# Custom F1-score metric
def f1_score(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    # Compute precision and recall manually
    true_positives = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(y_pred, tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(y_true, tf.float32))
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    
    # Compute F1-score
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    
    return f1

def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 64
    lr = 0.1
    num_epochs = 100
    model_path = os.path.join("files_UnetR/0930", "model.h5")
    csv_path = os.path.join("files_UnetR/0930", "log.csv")

    # """ RGB Code and Classes """
    # rgb_codes = [
    #     [0, 0, 0], [0, 1, 1], [2, 2, 2], 
    #     [3, 3, 3], [4, 4, 4], [5, 5, 5], 
    #     [6, 6, 6], [7, 7, 7], [8, 8, 8]
    # ]

    # classes = [
    #     "background", "annual_ryegrass", "bahia", "bermuda",
    #     "crabgrass", "brown top millet", "lespedeza", "johnsongrass", "others"
    # ]
    
    rgb_codes = [
        [0, 0, 0], [0, 1, 1], [2, 2, 2], 
        [3, 3, 3], [4, 4, 4], [5, 5, 5], 
        [6, 6, 6], [7, 7, 7]
    ]

    classes = [
        "background", "annual_ryegrass", "bahia", "bermuda",
        "crabgrass", "brown top millet", "lespedeza", "johnsongrass"
    ]

    """ Dataset """
    dataset_path = "PLSC_Upd"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    # (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    # print(f"Test: \t{len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_unetr_2d(cf)
    model.compile(loss="categorical_crossentropy",
                  metrics=['accuracy', 
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall'),
                            iou_score,
                            f1_score], 
                  optimizer=Adam(lr))

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=13, restore_best_weights=False)
    ]
    
    print(model.summary())

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        # callbacks=[tensorboard_callback]
        callbacks=callbacks
    )