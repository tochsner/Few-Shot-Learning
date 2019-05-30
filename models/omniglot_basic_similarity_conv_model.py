"""
Builds a simple convolutional nerual network for omniglot similarity (quadruplet encoder).
Output format of the Keras model: Embedding ; Output (Flatten)
Embedding-Length = 64
"""

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Concatenate, Activation, BatchNormalization
from keras.regularizers import l2


def build_model(input_shape, embedding_length):
    output_length = input_shape[0] * input_shape[1]
    input_layer = Input(shape=input_shape)
    conv = Conv2D(265, (10, 10), activation="relu")(input_layer)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D((2,2))(conv)
    conv = Conv2D(265, (7, 7), activation="relu")(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D((2,2))(conv)
    conv = Conv2D(265, (4, 4), activation="relu")(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D((2,2))(conv)
    conv = Conv2D(265, (4, 4), activation="relu")(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D((2,2))(conv)
    
    dense = Flatten()(conv)    

    encoder_output_layer = Dense(embedding_length, activation="sigmoid")(dense)

    decoder_dense = Dense(4096, activation='relu')(encoder_output_layer)    
    decoder_dense = Dropout(0.2)(decoder_dense)

    decoder_output_layer = Dense(output_length, activation='sigmoid')(encoder_output_layer)

    output_layer = Concatenate()([encoder_output_layer, decoder_output_layer])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model