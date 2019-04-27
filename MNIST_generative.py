"""
Trains a simple quadruplet cross-digit encoder on MNIST.
"""

from keras.optimizers import SGD
from data.MNIST import *
from helper.prepare_triplets import *
from models.MNIST_conv_similarity_model import *
from helper.losses_similarity import *

input_shape = (28, 28, 1)
input_lenght = 784
embedding_lenght = 400

epochs = 200
samples_per_epoch = 5000
batch_size = 20
number_test_samples = 2000
lr = 0.5

losses = Losses(input_lenght, embedding_lenght, decoder_factor=0.8)

data = load_data()
data = prepare_data_for_keras(data)
data_train = group_data(data[0])
data_test = group_data(data[1])


def train_model():
    model = build_model(input_shape, embedding_lenght)
    model.compile('adam', losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

    for e in range(epochs):
        for b in range(samples_per_epoch // batch_size):
            (x_train, y_train) = create_training_data_for_quadruplet_loss(model, data_train, batch_size, embedding_lenght)
            model.fit(x_train, y_train, epochs=1, verbose=0)

        (x_test, y_test) = create_training_data_for_quadruplet_loss(model, data_test, number_test_samples, embedding_lenght)
        print(model.evaluate(x_test, y_test, verbose=0)[1])

    (x_train, y_train) = create_training_data_for_quadruplet_loss(model, data_train, number_test_samples * 5,
                                                                  embedding_lenght)
    (x_test, y_test) = create_training_data_for_quadruplet_loss(model, data_test, number_test_samples, embedding_lenght)
    print("Model " + " Training-Accuracy:" + str(model.evaluate(x_train, y_train, verbose=0)[1]))
    print("Model " + " Test-Accuracy:" + str(model.evaluate(x_test, y_test, verbose=0)[1]))

    # save model
    model.save("saved_models/MNIST Generative Mine")


train_model()