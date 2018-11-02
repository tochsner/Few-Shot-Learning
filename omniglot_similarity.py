"""
Trains a simple quadruplet cross-digit encoder on ominglot.
"""

from data.omniglot import *
from helper.prepare_triplets import *
from models.omniglot_basic_similarity_model import *
from helper.losses_similarity import *

input_shape = (105, 105, 1)
input_length = 105 * 105
embedding_length = 40

epochs = 100
samples_per_epoch = 5000
batch_size = 20
number_test_samples = 2000

losses = Losses(input_length, embedding_length, decoder_factor=1)

data = load_background_data()
grouped_data = prepare_grouped_data_for_keras(data)
data_train, data_test = split_list(grouped_data, 0.7)


def train_model():
    model = build_model(input_shape, embedding_length)
    model.compile('adam', losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

    for e in range(epochs):
        for b in range(samples_per_epoch // batch_size):
            (x_train, y_train) = createTrainingDataForQuadrupletLoss(model, data_train, batch_size, embedding_length)
            model.fit(x_train, y_train, epochs=1, verbose=0)

        (x_test, y_test) = createTrainingDataForQuadrupletLoss(model, data_test, number_test_samples, embedding_length)
        print(model.evaluate(x_test, y_test, verbose=0)[1])


train_model()
