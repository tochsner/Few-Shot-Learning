"""
Trains a quadruplet cross-digit encoder on MNIST.
Uses talos for hyperparameter search.
"""

import os
import tensorflow as tf

import talos as ta
import numpy as np
from talos.model.normalizers import lr_normalizer

from keras.optimizers import SGD, Adam, Nadam, RMSprop, Adadelta
from keras.activations import relu, elu, tanh, sigmoid, selu

from data.MNIST import *
from helper.prepare_triplets import *
from models.MNIST_basic_similarity_model_talos import *
from helper.losses_similarity import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

input_shape = (28,28,1)
input_lenght = 784

data = load_data()
data = prepare_data_for_keras(data)
data_train = group_data(data[0])
data_test = group_data(data[1])

samples_per_epoch = 1000
number_test_samples = 4000 

def train(x_train, y_train, x_val, y_val, params):
    epochs = params['epochs']    
    batch_size = params['batch_size']     

    losses = Losses(input_lenght, params['embeddings'])

    model = build_model(input_shape, params)
    model.compile(params['optimizer'](), 
                    loss=losses.quadruplet_loss,
                    metrics=[losses.quadruplet_metric])
    
    for e in range(epochs):        
        for b in range(samples_per_epoch // batch_size):
            (x_train, y_train) = create_training_data_for_quadruplet_loss(model, data_train, batch_size, params['embeddings'])
            out = model.fit(x_train, y_train, epochs=1, verbose=0)
    
    (x_test, y_test) = create_training_data_for_quadruplet_loss(model, data_test, number_test_samples, params['embeddings'])
    
    out = model.fit(x_train, y_train, epochs=1, verbose=0, validation_data=(x_test, y_test))
    
    return out, model

params = {'lr':  [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'epochs': [50],
            'batch_size': [20],
            'optimizer': [Adam],
            'embeddings': [20],
            'activation': [relu],
            'neurons': [100],
            'decoderfactor': [2],
            'dropout': [0]}

x_test_fake = np.zeros((4000, 28, 28, 1))
y_test_fake = np.zeros((4000))
y_test_fake[0] = 9

h = ta.Scan(x_test_fake, y_test_fake, params=params,
            model=train,
            experiment_no='1',
            functional_model=True,
            grid_downsample=1)  
