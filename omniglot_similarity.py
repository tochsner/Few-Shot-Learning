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
number_test_samples = 2000

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

        (x_test, y_test) = createTrainingDataForQuadrupletLoss(model, data_test, number_test_samples, embedding_length)
        print(model.evaluate(x_test, y_test, verbose=0)[1], flush=True)

    model.save_weights("saved_models/omniglot_verification " + str(run_number))


for i in range(10):
    train_model(i)
