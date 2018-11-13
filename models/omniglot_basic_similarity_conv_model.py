"""
Builds a simple convolutional nerual network for omniglot similarity (quadruplet encoder).
Output format of the Keras model: Embedding ; Output (Flatten)
Embedding-Length = 64
"""

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate
from keras.regularizers import l2


def build_model(input_shape):
    output_length = input_shape[0] * input_shape[1]
    input_layer = Input(shape=input_shape)
    conv = Conv2D(64, (3, 3), activation="relu")(input_layer)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D((2, 2))(conv)
    conv = Conv2D(64, (3, 3), activation="relu")(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D((2, 2))(conv)
    conv = Conv2D(128, (3, 3), activation="relu")(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D((2, 2))(conv)
    flatten = Flatten()(conv)
    dense = Dense(256, activation="relu")(flatten)
    encoder_output_layer = Dense(32, activation="sigmoid")(dense)

    decoder_dense = Dense(128, activation='relu')(encoder_output_layer)
    decoder_dense = Dense(256, activation='relu')(decoder_dense)
    decoder_dense = Dense(512, activation='relu')(decoder_dense)
    decoder_output_layer = Dense(output_length, activation='sigmoid')(decoder_dense)

    output_layer = Concatenate()([encoder_output_layer, decoder_output_layer])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model