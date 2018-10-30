"""
Builds a simple convolutional nerual network for MNIST few shot (quadruplet encoder).
Output format of the Keras model: Embedding ; Output (Flatten)
"""

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Concatenate
from keras.regularizers import l2


def build_model(input_shape, embedding_dimensions):
    input_layer = Input(shape=input_shape)
    dense = Flatten()(input_layer)
    dense = Dense(100, activation='relu', kernel_regularizer=l2(0.0001))(dense)

    encoder_output_layer = Dense(embedding_dimensions, activation='sigmoid', kernel_regularizer=l2(0.0001))(dense)
    
    decoder_dense = Dense(100, activation='relu', kernel_regularizer=l2(0.0001))(encoder_output_layer)
    decoder_output_layer = Dense(784, activation='sigmoid')(decoder_dense)

    output_layer = Concatenate()([encoder_output_layer, decoder_output_layer])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model