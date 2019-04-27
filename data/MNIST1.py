""" 
Downloads the MNIST dataset and prepares it for the use with keras and tooc.
"""

import keras
from keras import utils
from keras.datasets import mnist
import numpy as np
import random

spectrogram_path = "/home/tobia/Documents/ML/Data MA Sono"
spectrogram_type = ".png"
playlists_path = "C:/Users/tobia/Documents/Programmieren/AI/MusicSimilarity/data/Playlists.csv"
songfeatures_path = "C:/Users/tobia/Documents/Programmieren/AI/MusicSimilarity/data/songfeatures small.csv"

"""
Imports a list of the songs in each playlist.
"""
def load_playlists():
    with open(playlists_path, "r") as f:
        return [np.array(x.strip().split(','))[2:] for x in f.readlines()]

def load_songfeatures_grouped():
    playlists = load_playlists()

    features = {}

    with open(songfeatures_path, "r") as f:
        for x in f.readlines():
            try:
                a = x.strip().split(',')                
                features[a[1]] = np.array(a[2:], dtype=np.float).reshape((1, 276, 1)) 
                features[a[1]] = np.maximum(0, np.minimum(np.power(features[a[1]], 1/2), 1))
            except:
                pass
        print(len(features)) 

    songfeatures_grouped = [[] for p in range(len(playlists))]

    for i, playlist in enumerate(playlists):
        for song in playlist:
            try:            
                songfeatures_grouped[i].append(np.array(features[song]).reshape((276,1,1)))
            except:
                pass
    
    random.shuffle(songfeatures_grouped)

    data_train = songfeatures_grouped[:int(0.7 * len(songfeatures_grouped))]
    data_test = songfeatures_grouped[int(0.7 * len(songfeatures_grouped)):]

    return data_train, data_test

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

"""
Formats the data for classification with my own implementation of neural nets.
"""
def prepare_data_for_tooc(data):
    (x_train, y_train), (x_test, y_test) = data

    # uses channels_last
    x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return ((x_train, y_train), (x_test, y_test))