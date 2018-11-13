"""
Builds a simple convolutional nerual network for MNIST similarity (quadruplet encoder).
Output format of the Keras model: Embedding ; Output (Flatten)
"""

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Concatenate
from keras.regularizers import l2

def build_model(input_shape, embedding_dimensions):
    input_layer = Input(shape=input_shape)
    dense = Flatten()(input_layer)
    dense = Dense(200, activation='relu')(dense)
    dense = Dropout(0.25)(dense)
    dense = Dense(200, activation='relu')(dense)

    encoder_output_layer = Dense(embedding_dimensions, activation='sigmoid')(dense)
    
    decoder_dense = Dense(200, activation='relu')(encoder_output_layer)
    decoder_dense = Dropout(0.25)(decoder_dense)
    decoder_dense = Dense(200, activation='relu')(decoder_dense)
    decoder_output_layer = Dense(784, activation='sigmoid')(decoder_dense)    

    output_layer = Concatenate()([encoder_output_layer, decoder_output_layer])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model