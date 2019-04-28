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

epochs = 5000
samples_per_epoch = 5000
test_tasks = 750

data = load_background_data()
data_train, data_test = split_list(data, 0.7)
data_train, data_test = prepare_grouped_data_for_keras(data_train), prepare_grouped_data_for_keras(data_test)

"""
Converts the continous parameters to discrete values and calls train_model.
"""
def train_wrapper(lr):
    batch_size = 32 # 2**int(log_batch_size)
    embedding_length = 64 # 2**int(log_embedding_length)
    momentum = 0.99
    decoder_factor = 0
    return train_model(lr, momentum, decoder_factor, batch_size, embedding_length)

def train_model(lr, momentum, decoder_factor, batch_size, embedding_length):    
    losses = Losses(input_length, embedding_length, decoder_factor)

    optimizer = Adam(lr)

    model = build_model(input_shape, embedding_length)
    model.compile(optimizer, losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

    print("Start with", lr, momentum, decoder_factor, batch_size, embedding_length)

    for e in range(epochs):
        for b in range(samples_per_epoch // batch_size):
            characters = random.choice(data_train)
            (x_train, y_train) = create_training_data_for_quadruplet_loss(model, characters, batch_size, embedding_length)        

            model.fit(x_train, y_train, epochs=1, verbose=0)

        verification_accuracy = calculate_verification_accuracy(model, data_test, embedding_length)
        oneshot_accuracy = calculate_20_way_1_shot_accuracy(model, embedding_length)

        print(e, ":", verification_accuracy, oneshot_accuracy)

    verification_accuracy = calculate_verification_accuracy(model, data_test, embedding_length)
    oneshot_accuracy = calculate_20_way_1_shot_accuracy(model, embedding_length)
    print(verification_accuracy, oneshot_accuracy)

    return oneshot_accuracy


def calculate_verification_accuracy(model, data_test, embedding_length):
    batch_size = 32
    trials = 1000
    
    accuracy = 0    
    count = 0
    for b in range(trials // batch_size):
        characters = random.choice(data_test)
        (x_train, y_train) = create_training_data_for_quadruplet_loss(model, characters, batch_size, 
                                                                        embedding_length, mode=1)
        accuracy += model.evaluate(x_train, y_train, verbose=0)[1]
        count += 1

    return accuracy / count

def calculate_20_way_1_shot_accuracy(model, embedding_length):
    n = 20
    k = 1
   
    tests = 0
    accuracy = 0

    for t in range(test_tasks):                
        prototypes = []
        data = []

        characters = random.choice(data_test)
        data = sample_data_for_n_way_k_shot(characters, n, k + 1)
        query_character = data[0][0]
        support_set = [x[1:] for x in data]

        query_embedding = get_embedding(model.predict(np.array(query_character).reshape(1,28,28,1)), embedding_length)

        for i in range(n):
            predictions = get_embedding(model.predict(np.array(support_set[i])), embedding_length)
            prototypes.append(np.mean(predictions, axis=0))
                      
        distances = [losses.get_distance(prototype, query_embedding) for prototype in prototypes]
        predicted_index = np.argmin(distances)
        if predicted_index == 0:
            accuracy += 1
        tests += 1

    return accuracy / tests

train_model(1e-6, 0.99, 0, 32, 64)

parameter_bounds = {'lr': (1e-7, 1e-3)}

optimizer = BayesianOptimization(   f=train_wrapper,
                                    pbounds=parameter_bounds,
                                    random_state=1
                                )

optimizer.maximize(init_points=15, n_iter=100)