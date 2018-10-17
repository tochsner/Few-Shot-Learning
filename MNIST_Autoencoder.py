from data.MNIST import *
from models.MNIST_Similarity import *
from helper.hyperparameter import *
from helper.prepareTriplets import *

def train_model():
    data = load_data()
    data = prepare_data_for_keras(data)
    training_data = group_data(data[0])

    hp = Hyperparameter()
    hp.epochs = 50

    model = build_model()

    model.compile(loss=hp.loss,
                optimizer=hp.optimizer,
                metrics=hp.metrics) 

    createTrainingDataForQuadrupletLoss(model, training_data, 100, 20)    

train_model()