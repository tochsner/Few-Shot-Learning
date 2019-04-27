"""
Trains a simple quadruplet cross-digit encoder on ominglot.
"""

from data.omniglot import *
from helper.prepare_triplets import *
from models.omniglot_basic_similarity_conv_model import *
from helper.losses_similarity import *
from keras.optimizers import Adam, SGD
import itertools

input_shape = (28, 28, 1)
input_length = 28, 28
embedding_length = 32

epochs = 30
samples_per_epoch = 5000
batch_size = 32
test_tasks_per_epoch = 1000

lr = 0.002
momentum = 0.5

losses = Losses(input_length, embedding_length, decoder_factor=0.0)

data = load_background_data()
data_train, data_test = split_list(data, 0.7)
data_train, data_test = prepare_grouped_data_for_keras(data_train), prepare_grouped_data_for_keras(data_test)

print(sum([len(x) for x in data_train]))

def train_model(run_number=0):    
    optimizer = SGD(lr, momentum=momentum)

    model = build_model(input_shape)
    model.compile(optimizer, losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

    model.summary()

    for e in range(epochs):
        train_accuracy = 0
        train_tests = 0        
        
        for b in range(samples_per_epoch // batch_size):
            characters = random.choice(data_train) # list(itertools.chain.from_iterable(data_train))
            (x_train, y_train) = create_training_data_for_quadruplet_loss(model, characters, batch_size, embedding_length)        

            model.fit(x_train, y_train, epochs=1, verbose=0)

        # evaluate test accuarcy
        test_accuracy = 0
        test_tests = 0
        for b in range(test_tasks_per_epoch // batch_size):
            characters = random.choice(data_test)
            (x_train, y_train) = create_training_data_for_quadruplet_loss(model, characters, batch_size, embedding_length, mode=0)
            test_accuracy += model.evaluate(x_train, y_train, verbose=0)[1]
            test_tests += 1

        print(test_accuracy / test_tests)
        
    model.save_weights("saved_models/omniglot_verification " + str(run_number))


def calculate_k_shot_accuracy(model, n=20):
    k = 1    
    
    tests = 0
    accuracy = 0

    for t in range(test_tasks_per_epoch):                
        prototypes = []
        test_data = []

        while len(test_data) < n: 
            characters = random.choice(data_test)
            test_data = sample_data_for_n_way_k_shot(characters, n, k + 15)
        
        support_set = [x[:k] for x in test_data]
        query_set = [x[k:] for x in test_data]

        for i in range(n):
            predictions = get_embedding(model.predict(np.array(support_set[i])), embedding_length)
            prototypes.append(np.mean(predictions, axis=0))                        

        for i in range(n):
            query_predictions = get_embedding(model.predict(np.array(query_set[i])), embedding_length)            

            for j in range(15):                
                distances = [losses.get_distance(prototypes[c], query_predictions[j]) for c in range(n)]
                predicted_index = np.argmin(distances)
                if predicted_index == i:
                    accuracy += 1
                tests += 1

    return accuracy / max(1, tests)

train_model()