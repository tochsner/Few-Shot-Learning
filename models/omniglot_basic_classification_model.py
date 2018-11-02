#builds a simple convolutional nerual network for omniglot classification

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def build_model():
    input_shape = (105, 105, 1) #channels last
    num_classes = 964

    input_layer = Input(shape=input_shape)
    dense = Flatten()(input_layer)
    dense = Dense(200, activation='relu')(dense)
    dense = Dropout(0.25)(dense)
    dense = Dense(200, activation='relu')(dense)
    output_layer = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model