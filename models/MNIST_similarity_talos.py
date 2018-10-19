"""
Builds a simple convolutional nerual network for MNIST similarity (quadruplet encoder).
Output format of the Keras model: Embedding ; Output (Flatten)
Needs to be trained with talos.
"""

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Concatenate

def build_model(input_shape, params):
    input_layer = Input(shape=input_shape)    
    dense = Flatten()(input_layer)
    dense = Dense(params['neurons'], activation=params['activation'])(dense)
    dense = Dense(params['neurons'], activation=params['activation'])(dense)
    
    encoder_output_layer = Dense(params['embeddings'], activation='sigmoid')(dense)
    
    decoder_dense = Dense(params['neurons'], activation=params['activation'])(encoder_output_layer)    
    decoder_dense = Dense(params['neurons'], activation=params['activation'])(decoder_dense)    
    decoder_dense = Dense(784, activation='sigmoid')(decoder_dense)    
    decoder_output_layer = Reshape(input_shape)(decoder_dense)    

    flattened_Output = Flatten()(decoder_output_layer)
    output_layer = Concatenate()([encoder_output_layer, flattened_Output])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model