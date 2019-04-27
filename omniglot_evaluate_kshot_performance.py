"""
Determines the verification-performance on the evaluation-set of omniglot.
"""

from models.omniglot_basic_similarity_conv_model import *
from data.omniglot import *
from helper.prepare_triplets import *
from helper.losses_similarity import *

input_shape = (28, 28, 1)
input_length = 28 * 28
embedding_length = 32


def calculate_k_shot_accuracy(model, n, k):
    number_of_trials = 2000

    queries_per_trial = 15

    data = load_evaluation_data()
    grouped_data = prepare_grouped_data_for_keras(data)

    tests = 0
    accuracy = 0
    for t in range(number_of_trials):
        print(t, accuracy / max(1, tests))
        prototypes = []

        test_data = []

        while len(test_data) < n:
            characters = random.choice(grouped_data)
            test_data = sample_data_for_n_way_k_shot(characters, n, k + queries_per_trial)
        
        support_set = [x[:k] for x in test_data]
        query_set = [x[k:] for x in test_data]

        for c in range(n):            
            predictions = get_embedding(model.predict(np.array(support_set[c])), embedding_length)
            prototypes.append(np.mean(predictions, axis=0))

        for c in range(n):
            for q in range(queries_per_trial):
                query_prediction = get_embedding(model.predict(np.array(query_set[c][q].reshape((1, 28, 28, 1)))), embedding_length)

                distances = [losses.get_distance(prototypes[c], query_prediction) for c in range(n)]
                predicted_index = np.argmin(distances)
                if predicted_index == c:
                    accuracy += 1
                tests += 1

    return accuracy / max(1, tests)

model = build_model(input_shape)
model.load_weights("saved_models/omniglot_verification 0")

print(calculate_k_shot_accuracy(model, 5, 1))
