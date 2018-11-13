"""
Determines the verification-performance on the evaluation-set of omniglot.
"""

from models.omniglot_basic_similarity_conv_model import *
from data.omniglot import *
from helper.prepare_triplets import *
from helper.losses_similarity import *

input_shape = (35, 35, 1)
input_length = 35 * 35
embedding_length = 32


def evaluate(model):
    num_trios = 5000

    losses = Losses(input_length, embedding_length, decoder_factor=0.75)

    data = load_evaluation_data()
    grouped_data = prepare_grouped_data_for_keras(data)

    trios = createTrios(grouped_data, num_trios)

    accuracy = 0

    for trio in trios:
        outputs = model.predict_on_batch(trio)

        if losses.get_distance(outputs[0], outputs[1]) < losses.get_distance(outputs[0], outputs[2]):
            accuracy += 1

    return accuracy / num_trios


model = build_model(input_shape)
model.load_weights("saved_models/omniglot_verification")

print(evaluate(model))