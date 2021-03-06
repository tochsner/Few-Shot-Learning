import numpy as np
import random
from scipy.misc import imsave

from .losses_similarity import *

losses = Losses()

"""
Returns the dataset grouped after the different labels. returns (labels, num_samples_per_class, x) (as nested lists, x is an array)
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
Samples a list of trios of two in-label samples and one out-label sample.
"""
def createTrios(grouped_data, num_samples):
    num_classes = len(grouped_data)

    indexes = list(range(num_classes))

    trios = []

    for sample in range(num_samples):
        main_index = random.choice(indexes)
        second_index = random.choice([index for index in indexes if index != main_index])

        main_sample1 = random.choice(grouped_data[main_index])
        main_sample2 = random.choice(grouped_data[main_index])
        second_sample = random.choice(grouped_data[second_index])

        trio = np.array([main_sample1, main_sample2, second_sample])

        trios.append(trio)

    return trios

"""
Generates input and output pairs for performing similarity learning with Keras.
Based on quadruplet-selection.
Output format of the Keras model: Embedding ; Output (Flatten)
Format of y_train: Target Embedding ; Dissimilar Embedding ; Target Decoder Output
"""
def create_training_data_for_quadruplet_loss(model, grouped_data, num_samples, embedding_lenght, mode=0):
    num_classes = len(grouped_data)
    input_lenght = np.prod(grouped_data[0][0].shape)

    indexes = list(range(num_classes))
    
    x_shape = grouped_data[0][0].shape
    x_shape = (num_samples,) + x_shape

    y_shape = (num_samples, 2 * embedding_lenght + input_lenght)    

    x_data = np.zeros(x_shape)
    y_data = np.zeros(y_shape)

    for sample in range(num_samples // 2):
        exception = True

        while exception:
            try:
                main_index = random.choice(indexes)
                second_index = random.choice([index for index in indexes if index != main_index])

                main_sample1 = random.choice(grouped_data[main_index])
                main_sample2 = random.choice(grouped_data[main_index])
                second_sample1 = random.choice(grouped_data[second_index])
                second_sample2 = random.choice(grouped_data[second_index])
                exception = False
            except:
                pass

        outputs = model.predict(np.array([main_sample1, main_sample2, second_sample1, second_sample2]))

        costs = (losses.get_distance(outputs[0][ : embedding_lenght], outputs[2][ : embedding_lenght]),
                 losses.get_distance(outputs[0][ : embedding_lenght], outputs[3][ : embedding_lenght]),
                 losses.get_distance(outputs[1][ : embedding_lenght], outputs[2][ : embedding_lenght]),
                 losses.get_distance(outputs[1][ : embedding_lenght], outputs[3][ : embedding_lenght]))

        argmin = np.argmin(costs)

        if mode == 1:
            argmin = 0

        if argmin == 0:
            # mainSample 1
            x_data[2 * sample] = main_sample1    
            y_data[2 * sample, : embedding_lenght] = outputs[1][ : embedding_lenght]
            y_data[2 * sample, embedding_lenght : 2*embedding_lenght] = outputs[2][ : embedding_lenght]
            y_data[2 * sample, 2*embedding_lenght : ] = main_sample2.reshape((input_lenght,))

            # secondSample 1
            x_data[2 * sample + 1] = second_sample1           
            y_data[2 * sample + 1, : embedding_lenght] = outputs[3][ : embedding_lenght]
            y_data[2 * sample + 1, embedding_lenght : 2*embedding_lenght] = outputs[0][ : embedding_lenght]
            y_data[2 * sample + 1, 2*embedding_lenght : ] = second_sample2.reshape((input_lenght,))

        elif argmin == 1:
            # mainSample 1
            x_data[2 * sample] = main_sample1    
            y_data[2 * sample, : embedding_lenght] = outputs[1][ : embedding_lenght]
            y_data[2 * sample, embedding_lenght : 2*embedding_lenght] = outputs[3][ : embedding_lenght]
            y_data[2 * sample, 2*embedding_lenght : ] = main_sample2.reshape((input_lenght,))

            # secondSample 2
            x_data[2 * sample + 1] = second_sample2            
            y_data[2 * sample + 1, : embedding_lenght] = outputs[2][ : embedding_lenght]
            y_data[2 * sample + 1, embedding_lenght : 2*embedding_lenght] = outputs[0][ : embedding_lenght]
            y_data[2 * sample + 1, 2*embedding_lenght : ] = second_sample1.reshape((input_lenght,))
        elif argmin == 2:
            # mainSample 2
            x_data[2 * sample] = main_sample2   
            y_data[2 * sample, : embedding_lenght] = outputs[0][ : embedding_lenght]
            y_data[2 * sample, embedding_lenght : 2*embedding_lenght] = outputs[2][ : embedding_lenght]
            y_data[2 * sample, 2*embedding_lenght : ] = main_sample1.reshape((input_lenght,))

            # secondSample 1
            x_data[2 * sample + 1] = second_sample1
            y_data[2 * sample + 1, : embedding_lenght] = outputs[3][ : embedding_lenght]
            y_data[2 * sample + 1, embedding_lenght : 2*embedding_lenght] = outputs[1][ : embedding_lenght]
            y_data[2 * sample + 1, 2*embedding_lenght : ] = second_sample2.reshape((input_lenght,))
        elif argmin == 3:
            # mainSample 2
            x_data[2 * sample] = main_sample2   
            y_data[2 * sample, : embedding_lenght] = outputs[0][ : embedding_lenght]
            y_data[2 * sample, embedding_lenght : 2*embedding_lenght] = outputs[3][ : embedding_lenght]
            y_data[2 * sample, 2*embedding_lenght : ] = main_sample1.reshape((input_lenght,))

            # secondSample 2
            x_data[2 * sample + 1] = second_sample2
            y_data[2 * sample + 1, : embedding_lenght] = outputs[2][ : embedding_lenght]
            y_data[2 * sample + 1, embedding_lenght : 2*embedding_lenght] = outputs[1][ : embedding_lenght]
            y_data[2 * sample + 1, 2*embedding_lenght : ] = second_sample1.reshape((input_lenght,))

    return (x_data, y_data)

"""
Randomly chooses k samples from n classes for few-shot-learning
"""
def sample_data_for_n_way_k_shot(grouped_data, n, k):
    num_classes = list(range(len(grouped_data)))
    
    query_class = random.choice(num_classes)
    num_classes.remove(query_class)
    
    while len(num_classes) < n:
        num_classes += num_classes

    random.shuffle(num_classes)    

    sampled_grouped_data = []
    
    np.random.shuffle(grouped_data[query_class])
    sampled_grouped_data.append(grouped_data[query_class][:k])

    for c in num_classes[:n-1]:
        np.random.shuffle(grouped_data[c])
        sampled_grouped_data.append(grouped_data[c][:k])

    return sampled_grouped_data

"""
Returns the embedding of an output.
Output format of the Keras model: Embedding ; Output (Flatten)
"""
def get_embedding(output, embedding_lenght):
    if len(output.shape) == 1:
        return output[:embedding_lenght]
    else:
        return output[:,:embedding_lenght]

"""
Returns the decoder output.
Output format of the Keras model: Embedding ; Output (Flatten)
"""
def get_decoder_output(output, embedding_lenght):
    if len(output.shape) == 1:
        return output[embedding_lenght:]
    else:
        return output[:, embedding_lenght:]

"""
Splits the list into two sublists with ratio r.
"""
def split_list(original_list, r):
    number_samples_total = len(original_list)
    number_samples_1 = int(number_samples_total * r)

    return original_list[:number_samples_1], original_list[number_samples_1:]