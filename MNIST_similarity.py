"""
Trains a simple quadruplet cross-digit encoder on MNIST.
"""

from keras.optimizers import SGD
from data.MNIST import *
from helper.prepare_triplets import *
from models.MNIST_basic_similarity_model import *
from helper.losses_similarity import *
from keras.models import load_model

input_shape = (28,28,1)
input_lenght = 784
embedding_lenght = 40

epochs = 100
samples_per_epoch = 5000
batch_size = 20
number_test_samples = 2000
lr = 0.5

losses = Losses(input_lenght, embedding_lenght, decoder_factor=0.6)

data = load_data()
data = prepare_data_for_keras(data)
data_train = group_data(data[0])
data_test = group_data(data[1])

def trainModel():
    model = build_model(input_shape, embedding_lenght)
    model.compile('adam', losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

    for e in range(epochs):
        for b in range(samples_per_epoch // batch_size):
            (x_train, y_train) = create_training_data_for_quadruplet_loss(model, data_train, batch_size, embedding_lenght)
            model.fit(x_train, y_train, epochs=1, verbose=0)
            
        (x_test, y_test) = create_training_data_for_quadruplet_loss(model, data_test, number_test_samples, embedding_lenght)
        print(model.evaluate(x_test, y_test, verbose=0)[1])

    (x_train, y_train) = create_training_data_for_quadruplet_loss(model, data_train, number_test_samples * 5, embedding_lenght)
    (x_test, y_test) = create_training_data_for_quadruplet_loss(model, data_test, number_test_samples, embedding_lenght)
    print("Model " + str(r) + " Training-Accuracy:" + str(model.evaluate(x_train, y_train, verbose=0)[1]))
    print("Model " + str(r) + " Test-Accuracy:" + str(model.evaluate(x_test, y_test, verbose=0)[1]))

    #save model
    model.save("saved_models/MNIST Similarity Mine " + str(r))

for i in range(10):
    trainModel()
