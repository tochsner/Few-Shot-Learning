"""
Determines the 20-way one-shot performance on the evaluation-set of omniglot.
"""

from models.omniglot_basic_similarity_conv_model import *
from data.omniglot import *
from helper.prepare_triplets import *
from helper.losses_similarity import *
from scipy.misc import imsave

input_shape = (105, 105, 1)
input_length = 105 * 105
embedding_length = 1024

"""
Calculates the performance in a 20-way 1-shot task.
"""
def calculate_20_way_1_shot_accuracy(model, embedding_length, dataset):
    test_tasks = 3000
    n = 2
    k = 1
   
    tests = 0
    accuracy = 0

    for t in range(test_tasks):                
        prototypes = []
        data = []

        characters = random.choice(dataset)
        data = sample_data_for_n_way_k_shot(characters, n, k + 1)
        query_character = data[0][0]
        support_set = [x[1:] for x in data]

        query_embedding = get_embedding(model.predict(np.array(query_character).reshape(1,105,105,1)), embedding_length)

        for i in range(n):
            predictions = get_embedding(model.predict(np.array(support_set[i])), embedding_length)
            prototypes +=[predictions[0]]
        
        distances = [losses.get_distance(prototype, query_embedding) for prototype in prototypes]

        predicted_index = np.argmin(distances)

        if predicted_index == 0:
            accuracy += 1
        tests += 1

        if t % 10 == 0:
            print(accuracy / tests)

    return accuracy / tests

data_test = load_evaluation_data()
data_test = prepare_grouped_data_for_keras(data_test)

model = build_model(input_shape, embedding_length)
model.load_weights("saved_models/omniglot_verification 1")

print(calculate_20_way_1_shot_accuracy(model, embedding_length, data_test))
