"""
Trains a simple quadruplet cross-digit encoder on MNIST.
"""

from data.MNIST import *
from helper.prepareTriplets import *
from models.MNIST_Similarity import *
from helper.losses_similarity import *

input_shape = (28,28,1)
input_lenght = 784
embedding_lenght = 20

epochs = 100
samples_per_epoch = 5000
batch_size = 30
number_test_samples = 2000

losses = Losses(input_lenght, embedding_lenght)

data = load_data()
data = prepare_data_for_keras(data)
data_train = group_data(data[0])
data_test = group_data(data[1])

model = build_model(input_shape, embedding_lenght)
model.compile('adam', losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

for e in range(epochs):
    for b in range(samples_per_epoch // batch_size):
        (x_train, y_train) = createTrainingDataForQuadrupletLoss(model, data_train, batch_size, embedding_lenght)
        model.fit(x_train, y_train, epochs=1, verbose=0)
        
    (x_train, y_train) = createTrainingDataForQuadrupletLoss(model, data_test, number_test_samples, embedding_lenght)
    print(model.evaluate(x_train, y_train, verbose=0)[1])
