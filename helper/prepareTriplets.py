import numpy as np
import random

from losses import *

"""
Returns the dataset grouped after the different labels. returns (num_labels, num_samples_per_class, x) (as nested lists, x is an array)
"""
def group_data(data):
    x_data, y_data = data
    
    num_labels = y_data.shape[1]
    num_samples = x_data.shape[0]

    grouped_data = [[] for label in range(num_labels)]

    for sample in range(num_samples):
        label = np.argmax(y_data[sample])

        grouped_data[label].append(x_data[sample])

    return grouped_data



"""
Generates input and output pairs for performing similarity leraning with Keras.
Based on quadruplet-selection.
Output format of the Keras model: Input ; Output ; Embedding (Flatten)
"""
def createTrainingDataForQuadrupletLoss(model, grouped_data, num_samples, embedding_lenght):  
    num_classes = len(grouped_data)
    input_lenght = np.prod(grouped_data[0][0].shape)

    indexes = list(range(num_classes))

    print("J")
    
    x_shape = grouped_data[0][0].shape
    x_shape = (num_samples,) + x_shape

    y_shape = (num_samples, 2 * input_lenght + embedding_lenght, 1)    

    x_data = np.zeros(x_shape)
    y_data = np.zeros(y_shape)

    for sample in range(num_samples):
        mainIndex = random.choice(indexes)
        secondIndex = random.choice([index for index in indexes if index != mainIndex])

        mainSample1 = random.choice(grouped_data[mainIndex])
        mainSample2 = random.choice(grouped_data[mainIndex])
        secondSample1 = random.choice(grouped_data[secondIndex])
        secondSample2 = random.choice(grouped_data[secondIndex])
        
        outputs = model.predict(np.array([mainSample1, mainSample2, secondSample1, secondSample2]))

        