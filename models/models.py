import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.applications import ResNet50V2, InceptionV3, DenseNet121, VGG16
from params import CLASSES_NUM, INPUT_SHAPE

example_drop_out_rate = [0, 0.2, 0.4]
hidden_layers_activation_funcs = ['relu', 'selu', 'elu']



def original():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=INPUT_SHAPE))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', ))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
    model.add(MaxPool2D(pool_size=(2, 2))) # sprawdzic average pooling
    model.add(Dropout(0.40))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(CLASSES_NUM, activation='softmax'))
    model.summary()
    return model


def vgg_16():
    model = VGG16(
                include_top=False,
                input_shape=INPUT_SHAPE,
                )
    model.trainable = False
    inputs = keras.Input(INPUT_SHAPE)
    x = model(inputs, training=False)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(CLASSES_NUM)(x)
    model = keras.Model(inputs, outputs)
    return model


def resnet_50v2():
    model = ResNet50V2(
                    include_top=False,
                    input_shape=INPUT_SHAPE,
                    )
    model.trainable = False
    inputs = keras.Input(INPUT_SHAPE)
    x = model(inputs, training=False)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(CLASSES_NUM)(x)
    model = keras.Model(inputs, outputs)
    return model


def inception_v3():
    model = InceptionV3(
                        include_top=False,
                        input_shape=INPUT_SHAPE,
                        )
    model.trainable = False
    inputs = keras.Input(INPUT_SHAPE)
    x = model(inputs, training=False)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(CLASSES_NUM)(x)
    model = keras.Model(inputs, outputs)
    return model


def densenet_121():
    model = DenseNet121(
                        include_top=False,
                        input_shape=INPUT_SHAPE,
                        )
    model.trainable = False
    inputs = keras.Input(INPUT_SHAPE)
    x = model(inputs, training=False)
    x = keras.layers.GlobalAvgPool2D()(x)
    outputs = keras.layers.Dense(CLASSES_NUM)(x)
    model = keras.Model(inputs, outputs)
    return model


