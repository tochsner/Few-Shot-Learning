from data.MNIST import *
from helper.hyperparameter import *
from helper.NN import *
from helper.activations import *
from helper.losses import *
import numpy as np

def train_model():
    data = load_data()
    (x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)

    hp = Hyperparameter()
    hp.epochs = 1
    hp.lr = 0.2
    hp.r = 0.002

    mse = MeanSquareCostFunction()

    encoder = HebbianNetwork((784, 100, 10), np.tanh)
    classifier = SimpleNeuronalNetwork((784, 100, 10), sigmoidActivation, sigmoidDerivation, mse)

    print("train encoder...")

    for e in range(hp.epochs):
        for b in range(x_train.shape[0] // hp.batch_size):
            for s in range(hp.batch_size):
                encoder.trainNetwork(x_train[b * hp.batch_size + s, : ])
            encoder.applyChanges(hp.lr, hp.lr, hp.r)
            
        print("Epoch " + str(e) + "/" + str(hp.epochs), flush=True)
    
    print("train classifier...")

    hp.epochs = 20
    hp.lr = 1

    classifier.Weights[0] = encoder.Weights[0]

    for e in range(hp.epochs):
        for b in range(x_train.shape[0] // hp.batch_size):
            for s in range(hp.batch_size):
                classifier.trainNetwork(x_train[b * hp.batch_size + s, : ], y_train[b * hp.batch_size + s, : ])
            classifier.applyChanges(hp.lr, hp.lr, hp.r)

        accuracy = 0
        tests = 0

        for s in range(x_train.shape[0]):
            output = classifier.getOutput(x_train[s, : ])           
            if np.argmax(output) == np.argmax(y_train[s, : ]):
                accuracy += 1
            tests += 1

        print(accuracy / tests, flush = True)

train_model()