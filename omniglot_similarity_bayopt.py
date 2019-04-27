"""
Runs Bayesian Hyperparameter Optimization on a simple quadruplet cross-digit encoder on ominglot.
"""

from data.omniglot import *
from helper.prepare_triplets import *
from models.omniglot_basic_similarity_conv_model import *
from helper.losses_similarity import *
from keras.optimizers import Adam, SGD
import itertools
from bayes_opt import BayesianOptimization

input_shape = (28, 28, 1)
input_length = 28, 28

epochs = 1
samples_per_epoch = 3000
test_tasks = 3000

data = load_background_data()
data_train, data_test = split_list(data, 0.7)
data_train, data_test = prepare_grouped_data_for_keras(data_train), prepare_grouped_data_for_keras(data_test)

"""
Converts the continous parameters to discrete values and calls train_model.
"""
def train_wrapper(lr, momentum, decoder_factor, log_batch_size, log_embedding_length):
    batch_size = 2**int(log_batch_size)
    embedding_length = 2**int(log_embedding_length)
    train_model(lr, momentum, decoder_factor, batch_size, embedding_length)

def train_model(lr, momentum, decoder_factor, batch_size, embedding_length):    
    losses = Losses(input_length, embedding_length, decoder_factor)

    optimizer = SGD(lr, momentum)

    model = build_model(input_shape, embedding_length)
    model.compile(optimizer, losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

    print("Start with", lr, momentum, decoder_factor, batch_size, embedding_length)

    for e in range(epochs):
        for b in range(samples_per_epoch // batch_size):
            characters = random.choice(data_train)
            (x_train, y_train) = create_training_data_for_quadruplet_loss(model, characters, batch_size, embedding_length)        

            model.fit(x_train, y_train, epochs=1, verbose=0)

        print("Epoch", e, "done...")

    # evaluate test accuarcy
    test_accuracy = 0    
    for b in range(test_tasks // batch_size):
        characters = random.choice(data_test)
        (x_train, y_train) = create_training_data_for_quadruplet_loss(model, characters, batch_size, 
                                                                        embedding_length, mode=0)
        test_accuracy += model.evaluate(x_train, y_train, verbose=0)[1]

    print(test_accuracy / test_tasks)

    return test_accuracy / test_tasks


parameter_bounds = {'lr': (1e-5, 1),
                    'momentum': (0, 0.99),
                    'decoder_factor': (0, 1), 
                    'log_batch_size': (2, 8),
                    'log_embedding_length': (2, 7)}

optimizer = BayesianOptimization(   f=train_wrapper,
                                    pbounds=parameter_bounds,
                                    random_state=1
                                )

optimizer.maximize(init_points=10, n_iter=50)