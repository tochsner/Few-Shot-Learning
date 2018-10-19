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

def build_model(input_shape, embedding_dimensions):
    input_layer = Input(shape=input_shape)
    #conv = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    #conv = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv)
    #conv = MaxPooling2D(pool_size=(2, 2))(conv)
    #conv = Dropout(0.25)(conv)
    dense = Flatten()(input_layer)#(conv)
    dense = Dense(128, activation='relu')(dense)
    #dense = Dropout(0.5)(dense)
    encoder_output_layer = Dense(embedding_dimensions, activation='sigmoid')(dense)
    
    decoder_dense = Dense(128, activation='relu')(encoder_output_layer)    
    decoder_dense = Dense(784, activation='relu')(decoder_dense)    
    decoder_output_layer = Reshape(input_shape)(decoder_dense)    

    flattened_Output = Flatten()(decoder_output_layer)

    output_layer = Concatenate()([encoder_output_layer, flattened_Output])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model