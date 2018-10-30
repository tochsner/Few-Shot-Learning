"""
Downloads the omniglot dataset and prepares it for the use with keras.
"""

import keras
from keras import utils
from keras.datasets import mnist
import numpy as np


background_set_path = "C:/Users/tobia/Documents/Programmieren/AI/omniglot/images_background/images_background"
evaluation_set_path = "C:/Users/tobia/Documents/Programmieren/AI/omniglot/images_background/images_evaluation"

img_rows = 105
img_cols = 105

"""
Loads the omniglot dataset.
Format: ([language, character, writer, array(105, 105)], ([language_test, character, writer, array(105, 105)])
"""
def load_data():
    background_data = []
    evaluation_data = []



    return ((x_train, y_train), (x_test, y_test))

"""     
Formats the data for classification with Keras.
"""
def prepare_data_for_keras(data):
    (x_train, y_train), (x_test, y_test) = data

    # uses channels_last
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return ((x_train, y_train), (x_test, y_test))
