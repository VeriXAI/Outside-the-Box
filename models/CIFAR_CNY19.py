from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.keras.layers import Convolution2D, Dense, MaxPooling2D, Activation, BatchNormalization,\
                                                Dropout, Flatten


# Note: 'weights' is ignored and just present for compatibility with other networks
def CIFAR_CNY19(classes, input_shape, weights=None):
    model = Sequential()

    model.add(Convolution2D(40, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(20, (5, 5), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(240, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(84, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model
