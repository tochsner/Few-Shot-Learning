"""
Trains a simple quadruplet cross-digit encoder on ominglot. Uses hyperparameter search to find optimal
parameters.
"""

from data.omniglot import *
from helper.prepare_triplets import *
from models.omniglot_basic_similarity_conv_model import *
from helper.losses_similarity import *
from helper.hyperparameter_search import *
from keras.optimizers import nadam, adam, sgd, rmsprop

hp_search = HyperparameterSearch("Omniglot Similarity Conv")

input_shape = (28, 28, 1)
input_length = 784

epochs = 20
samples_per_epoch = 1000
number_test_samples = 200

data = load_background_data()
grouped_data = prepare_grouped_data_for_keras(data)
data_train, data_test = split_list(grouped_data, 0.7)


def train_model(param):
    embedding_length = 32
    batch_size = param["batch_size"]

    losses = Losses(input_length, embedding_length, decoder_factor=param["decoder_factor"])

    model = build_model(input_shape)
    model.compile(param["optimizer"](), losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

    for e in range(epochs):
        for b in range(samples_per_epoch // batch_size):
            characters = random.choice(data_train)
            (x_train, y_train) = create_training_data_for_quadruplet_loss(model, characters, batch_size, embedding_length)
            model.fit(x_train, y_train, epochs=1, verbose=0)    
    
    x_test, y_test = create_training_data_for_quadruplet_loss(model, data_test[0], number_test_samples, embedding_length)

    for i in range(1, len(data_test)):
        x_test_char, y_test_char = create_training_data_for_quadruplet_loss(model, data_test[0], number_test_samples, embedding_length)
        x_test, y_test = np.concatenate((x_test, x_test_char), axis=0), np.concatenate((y_test, y_test_char), axis=0)

    return model.evaluate(x_test, y_test, verbose=0)


parameter = {"batch_size": [30],
             "decoder_factor": [0.25, 0.5, 0.75],
             "optimizer": [adam, nadam, sgd, rmsprop],
             "lr": [0.0001, 0.003, 0.001, 0.003, 0.01, 0.03, 0.1]}

hp_search.scan(train_model, parameter, 1)