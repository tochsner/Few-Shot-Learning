"""
Trains a simple quadruplet cross-digit encoder on ominglot. Uses hyperparameter search to find optimal
parameters.
"""

from data.omniglot import *
from helper.prepare_triplets import *
from models.omniglot_basic_similarity_model import *
from helper.losses_similarity import *
from helper.hyperparameter_search import *

hp_search = HyperparameterSearch("Omniglot Similarity")

input_shape = (105, 105, 1)
input_length = 105 * 105

epochs = 20
samples_per_epoch = 1000
number_test_samples = 2000

data = load_background_data()
grouped_data = prepare_grouped_data_for_keras(data)
data_train, data_test = split_list(grouped_data, 0.7)


def train_model(param):
    embedding_length = param["embedding_length"]
    batch_size = param["batch_size"]

    losses = Losses(input_length, embedding_length, decoder_factor=param["decoder_factor"])

    model = build_model(input_shape, embedding_length)
    model.compile(param["optimizer"], losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

    for e in range(epochs):
        for b in range(samples_per_epoch // batch_size):
            (x_train, y_train) = createTrainingDataForQuadrupletLoss(model, data_train, batch_size, embedding_length)
            model.fit(x_train, y_train, epochs=1, verbose=0)

    (x_test, y_test) = createTrainingDataForQuadrupletLoss(model, data_test, number_test_samples, embedding_length)
    return model.evaluate(x_test, y_test, verbose=0)


parameter = {"batch_size": [20, 40],
             "embedding_length": [20, 40, 60],
             "decoder_factor": [0, 1, 2],
             "optimizer": ["Adagrad"]}

hp_search.scan(train_model, parameter, 1)