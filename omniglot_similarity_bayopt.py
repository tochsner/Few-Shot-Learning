"""
Runs Bayesian Hyperparameter Optimization on a quadruplet cross-digit encoder on ominglot.
"""

from data.omniglot import *
from helper.prepare_triplets import *
from models.omniglot_basic_similarity_model import *
from helper.losses_similarity import *
from keras.optimizers import Adam, SGD
from bayes_opt import BayesianOptimization

input_shape = (28, 28, 1)
input_length = 28, 28

epochs = 40
samples_per_epoch = 1000
test_tasks = 750

data = load_background_data()
data_train, data_test = split_list(data, 0.7)
data_train, data_test = prepare_grouped_data_for_keras(data_train), prepare_grouped_data_for_keras(data_test)


"""
Wrapper of train_model which gets called for bayesian optimization.
Enables fixing certain hyperparameters and converting the continous 
values provided by the optimization algorithm to discrete values.
"""
def train_wrapper(lr):
    batch_size = 64
    embedding_length = 1024
    momentum = 0.99 
    decoder_factor = 0    
    return train_model(lr, momentum, decoder_factor, batch_size, embedding_length)


"""
Trains a CNN on omniglot-verification using a quadruplet cross-character encoder.
Evaluates the model on verification and 20-way 1-shot tasks.
"""
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
        
        print(e)

    verification_accuracy = calculate_verification_accuracy(model, data_test, embedding_length)
    #oneshot_accuracy = calculate_20_way_1_shot_accuracy(model, embedding_length)
    print(verification_accuracy)

    return verification_accuracy


"""
Calculates the probability that two images of the same character are rated as more
similar than two distinct characters.
"""
def calculate_verification_accuracy(model, data_test, embedding_length):
    batch_size = 32
    trials = 5000
    
    accuracy = 0    
    count = 0

    for b in range(trials // batch_size):
        characters = random.choice(data_test)
        (x_train, y_train) = create_training_data_for_quadruplet_loss(model, characters, batch_size, embedding_length, mode=1)
        
        accuracy += model.evaluate(x_train, y_train, verbose=0)[1]
        count += 1

    return accuracy / count


"""
Calculates the performance in a 20-way 1-shot task.
"""
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

        # Sometimes, due to a bad choice of a learning rate, all embeddings are identical.
        # Because argmax is then misleading, we ignore this case.
        if max(distances) > 1e-5:
            predicted_index = np.argmin(distances)

            if predicted_index == 0:
                accuracy += 1
        tests += 1

    return accuracy / tests


parameter_bounds = {'lr': (1e-9, 1e-4)}

optimizer = BayesianOptimization(   f=train_wrapper,
                                    pbounds=parameter_bounds,
                                    random_state=1)

optimizer.maximize(init_points=20, n_iter=100)

print(optimizer.max)