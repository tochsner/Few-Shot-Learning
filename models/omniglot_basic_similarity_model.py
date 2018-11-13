"""
Builds a simple convolutional nerual network for omniglot similarity (quadruplet encoder).
Output format of the Keras model: Embedding ; Output (Flatten)
"""

import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Concatenate
from keras.regularizers import l2

def build_model(input_shape, embedding_dimensions):
    output_length = input_shape[0] * input_shape[1]
    input_layer = Input(shape=input_shape)
    dense = Flatten()(input_layer)
    dense = Dense(400, activation='relu')(dense)
    dense = Dense(400, activation='relu')(dense)

    encoder_output_layer = Dense(embedding_dimensions, activation='sigmoid')(dense)
    
    decoder_dense = Dense(400, activation='relu')(encoder_output_layer)
    decoder_dense = Dense(400, activation='relu')(decoder_dense)
    decoder_output_layer = Dense(output_length, activation='sigmoid')(decoder_dense)

    output_layer = Concatenate()([encoder_output_layer, decoder_output_layer])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model