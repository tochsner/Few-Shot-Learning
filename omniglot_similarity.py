"""
Trains a simple quadruplet cross-digit encoder on ominglot.
"""

from data.omniglot import *
from helper.prepare_triplets import *
from models.omniglot_basic_similarity_conv_model import *
from helper.losses_similarity import *
from keras.optimizers import nadam

input_shape = (35, 35, 1)
input_length = 35 * 35
embedding_length = 32

epochs = 300
samples_per_epoch = 1000
batch_size = 32
number_test_samples = 50

optimizer = nadam(0.001)

losses = Losses(input_length, embedding_length, decoder_factor=0.75)

data = load_background_data()
data_train, data_test = split_list(data, 0.8)
data_train, data_test = prepare_grouped_data_for_keras(data_train), prepare_grouped_data_for_keras(data_test)


def train_model(run_number=0):
    model = build_model(input_shape)
    model.compile(optimizer, losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

    for e in range(epochs):
        for b in range(samples_per_epoch // batch_size):
            (x_train, y_train) = createTrainingDataForQuadrupletLoss(model, data_train, batch_size, embedding_length)
            model.fit(x_train, y_train, epochs=1, verbose=0)

        print(calculate_k_shot_accuracy(model), flush=True)

    model.save_weights("saved_models/omniglot_verification " + str(run_number))


def calculate_k_shot_accuracy(model):
    n = 5
    k = 1
    tests = 0
    accuracy = 0
    for t in range(number_test_samples):
        prototypes = []
        test_data = sample_data_for_n_way_k_shot(data_test, n, k + 15)
        support_set = [x[:k] for x in test_data]
        query_set = [x[k:] for x in test_data]

        for i in range(n):
            predictions = get_embedding(model.predict(np.array(support_set[i])), embedding_length)
            prototypes.append(np.mean(predictions, axis=0))

        for i in range(n):
            query_predictions = get_embedding(model.predict(np.array(query_set[i])), embedding_length)

            for j in range(query_predictions.shape[0]):
                distances = [losses.get_distance(prototypes[c], query_predictions[j]) for c in range(n)]
                prediction_index = np.argmin(distances)
                if prediction_index == i:
                    accuracy += 1
                tests += 1

    return accuracy / max(1, tests)


for i in range(10):
    train_model(i)
