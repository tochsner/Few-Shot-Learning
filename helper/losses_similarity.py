"""
The custom losses for similarity with siamese networks with keras.
Output format of the Keras model: Embedding ; Decoder Output (Flatten)
Format of y_train: Target Embedding ; Dissimlilar Embedding ; Target Decoder Output
"""

from keras import backend as K
from .losses import *

class Losses():
    def __init__(self, input_lenght = 0, embedding_lenght = 0):
        self.mse = MeanSquareCostFunction()

        self.input_lenght = input_lenght
        self.embedding_lenght = embedding_lenght 

    def get_distance(self, output_values, correct_values):
        return self.mse.get_cost(output_values, correct_values)   

    def trio_loss(self, y_true, y_pred):            
        output_embedding = y_pred[:, : self.embedding_lenght]
        target_embedding = y_true[:, : self.embedding_lenght]
        dissimilar_embedding = y_true[:, self.embedding_lenght : 2*self.embedding_lenght]

        return  K.mean(K.square(output_embedding - target_embedding), axis=-1) - \
                K.mean(K.square(output_embedding - dissimilar_embedding), axis=-1)
    
    
    def quadruplet_loss(self, y_true, y_pred):            
        output_embedding = y_pred[:, : self.embedding_lenght]
        target_embedding = y_true[:, : self.embedding_lenght]
        dissimilar_embedding = y_true[:, self.embedding_lenght : 2*self.embedding_lenght]

        decoder_output = y_pred[:, self.embedding_lenght : ]
        target_decoder_output = y_true[:, 2*self.embedding_lenght : ]        
        
        return K.mean(K.square(decoder_output - target_decoder_output), axis=-1) +  K.mean(K.square(output_embedding - target_embedding), axis=-1) + \
                 K.mean(K.square(output_embedding) / 2 - output_embedding * dissimilar_embedding - K.abs(dissimilar_embedding - output_embedding), axis=-1)
    
    def quadruplet_metric(self, y_true, y_pred):            
        output_embedding = y_pred[:, : self.embedding_lenght]
        target_embedding = y_true[:, : self.embedding_lenght]
        dissimilar_embedding = y_true[:, self.embedding_lenght : 2*self.embedding_lenght]
        
        return K.mean(K.cast(K.less(K.mean(K.square(output_embedding - target_embedding), axis=-1), K.mean(K.square(output_embedding - dissimilar_embedding), axis=-1)), 'float16'))