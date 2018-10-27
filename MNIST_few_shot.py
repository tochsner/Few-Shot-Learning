"""
Trains a simple quadruplet cross-digit encoder on MNIST.
Evaluates the performance on k-shot-learning classification.
"""

from keras.optimizers import SGD
from data.MNIST import *
from helper.prepare_triplets import *
from models.MNIST_basic_similarity_model import *
from helper.losses_similarity import *
from keras.optimizers import Adam

k = 5

input_shape = (28,28,1)
input_lenght = 784
embedding_lenght = 40

epochs = 20
samples_per_epoch = 1000
batch_size = 20
number_test_samples = 2000

losses = Losses(input_lenght, embedding_lenght, decoder_factor=2)

data = load_data()
data = prepare_data_for_keras(data)
data_train = group_data(data[0])
data_train = sample_data_for_k_shot(data_train, k)
data_test = group_data(data[1])


def evaluate_classification_accuracy(model):
    prototypes = []
    for i in range(num_classes):
        predictions = model.predict(np.array(data_train[i]))[:, :embedding_lenght]
        prototypes.append(np.mean(predictions, axis=0))

    accuracy = 0
    tests = 0

    for i in range(num_classes):
        test_predictions = model.predict(np.array(data_test[i]))[:, :embedding_lenght]
        for j in range(test_predictions.shape[0]):
            distances = [losses.get_distance(prototypes[c], test_predictions[c]) for c in range(num_classes)]
            prediction_index = np.argmin(distances)
            if prediction_index == i:
                accuracy += 1
            tests += 1

    return accuracy / max(tests, 1)


def train(iterations):
    for r in range(iterations):
        model = build_model(input_shape, embedding_lenght)
        model.compile(Adam(lr=0.001), losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

        for e in range(epochs):
            for b in range(samples_per_epoch // batch_size):
                (x_train, y_train) = createTrainingDataForQuadrupletLoss(model, data_train, batch_size, embedding_lenght)
                model.fit(x_train, y_train, epochs=1, verbose=0)

            print(evaluate_classification_accuracy(model))

        (x_train, y_train) = createTrainingDataForQuadrupletLoss(model, data_train, number_test_samples * 5, embedding_lenght)
        (x_test, y_test) = createTrainingDataForQuadrupletLoss(model, data_test, number_test_samples, embedding_lenght)
        print(str(k) + "-shot: Model " + str(r) + " Training-Accuracy:" + str(model.evaluate(x_train, y_train, verbose=0)[1]))
        print(str(k) + "-shot: Model " + str(r) + " Test-Accuracy:" + str(model.evaluate(x_test, y_test, verbose=0)[1]))
        print(str(k) + "-shot: Model " + str(r) + " Classification-Accuracy " + str(evaluate_classification_accuracy(model)))


k = 5
train(30)
k = 10
train(30)
k = 100
train(30)