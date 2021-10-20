import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D
from tensorflow.keras import Model

# 2. functional approach: function that returns a model
def functional_model():
    my_input = Input(shape=(28,28,1))
    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(10, activation = 'softmax')(x)

    model = tensorflow.keras.Model(inputs = my_input, outputs = x)
    return model

# 3. tensorflow.keras.Model : Inherit from this class
class MyCustomModel(tensorflow.keras.Model):

    def __init__(self):
        super().__init__()

        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxPool1 = MaxPool2D()
        self.Batch1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3,3), activation = 'relu')
        self.maxPool2 = MaxPool2D()
        self.Batch2 = BatchNormalization()
        
        self.Global = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation = 'relu')
        self.dense2 = Dense(10, activation = 'softmax')

    def call(self, my_input):
        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxPool1(x)
        x = self.Batch1(x)

        x = self.conv3(x)
        x = self.maxPool2(x)
        x = self.Batch2(x)

        x = self.Global(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

def streetsigns_model(nbr_classes):
    my_input = Input(shape=(60,60,3))

    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)    
    
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)        

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(nbr_classes, activation='softmax')(x)

    return Model(inputs = my_input, outputs = x)

if __name__ == '__main__':
    model = streetsigns_model(10)
    model.summary()