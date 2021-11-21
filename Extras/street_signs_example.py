from tensorflow.keras import callbacks
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.backend import categorical_crossentropy
from my_utils import order_test_set, split_data, create_generators
from deep_learning_models import streetsigns_model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

if __name__ == '__main__':
 
    if False:
        path_to_data = 'C:\\Users\\psingh24\\PythonClass\\Python practise\\German_Dataset\\Train'
        path_to_save_train = 'C:\\Users\\psingh24\\PythonClass\\Python practise\\German_Dataset\\training_data\\train'
        path_to_save_val = 'C:\\Users\\psingh24\\PythonClass\\Python practise\\German_Dataset\\training_data\\val'

        split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)

    # path_to_images = 'C:\\Users\\psingh24\\PythonClass\\Python practise\\German_Dataset\\Test'
    # path_to_csv = 'C:\\Users\\psingh24\\PythonClass\\Python practise\\German_Dataset\\Test.csv'
    # order_test_set(path_to_images, path_to_csv)

    path_to_train = 'C:\\Users\\psingh24\\PythonClass\\Python practise\\German_Dataset\\training_data\\train'
    path_to_val = 'C:\\Users\\psingh24\\PythonClass\\Python practise\\German_Dataset\\training_data\\val'
    path_to_test = 'C:\\Users\\psingh24\\PythonClass\\Python practise\\German_Dataset\\Test'
    batch_size = 64
    epochs = 15
    lr = 0.0001

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    TRAIN = False
    TEST = True

    if TRAIN:
        path_to_save_model = './Models'
        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            mode = 'max',
            monitor='val_accuracy',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=10
        )

        model = streetsigns_model(nbr_classes)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
        model.compile(optimizer=optimizer, metrics=['accuracy'], loss=categorical_crossentropy)

        model.fit(
            train_generator,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks = [ckpt_saver, early_stop]
        )
    
    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print('Evaluating validation set')
        model.evaluate(val_generator)

        print('Evaluating test set')
        model.evaluate(test_generator)