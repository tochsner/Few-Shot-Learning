"""
Downloads the omniglot dataset and prepares it for the use with keras.
"""

import os
import numpy as np
from PIL import Image

background_set_path = "data/omniglot/images_background/images_background"
evaluation_set_path = "data/omniglot/images_evaluation/images_evaluation"

img_rows = 28
img_cols = 28

def load_image(path):
    return 1 - np.array(Image.open(path).resize((img_rows, img_cols))) / 256
    

"""
Loads the omniglot dataset.
Format: [language, character, writer, array(105, 105)]
"""
def load_background_data():
    background_data = []

    for language in os.listdir(background_set_path):
        language_path = os.path.join(background_set_path, language)
        background_data.append([])
        for character in os.listdir(language_path):
            character_path = os.path.join(language_path, character)
            background_data[-1].append([]) # Create for every possible rotation a character
            background_data[-1].append([])
            background_data[-1].append([])
            background_data[-1].append([])
            for image in os.listdir(character_path):
                image_path = os.path.join(character_path, image)
                image = load_image(image_path)
                background_data[-1][-4].append(image)
                background_data[-1][-3].append(np.rot90(image, 1))
                background_data[-1][-2].append(np.rot90(image, 2))
                background_data[-1][-1].append(np.rot90(image, 3))                

    return background_data

def load_evaluation_data():
    evaluation_data = []

    for language in os.listdir(evaluation_set_path):
        language_path = os.path.join(evaluation_set_path, language)
        evaluation_data.append([])
        for character in os.listdir(language_path):
            character_path = os.path.join(language_path, character)
            evaluation_data[-1].append([])
            for image in os.listdir(character_path):
                image_path = os.path.join(character_path, image)
                image = load_image(image_path)
                evaluation_data[-1][-1].append(image)

    return evaluation_data

"""     
Formats the data for classification with Keras.
Format of data: [language, character, writer, array(105, 105)]
"""
def prepare_data_for_keras(data):
    num_characters = sum([len(lang) for lang in data])
    num_samples = sum([len(char) for lang in data for char in lang])

    # uses channels_last
    train = np.zeros((num_samples, img_rows, img_cols, 1))
    test = np.zeros((num_samples, num_characters))

    character_id = 0
    sample_id = 0
    for language in data:
        for character in language:
            for image in character:
                train[sample_id] = image.reshape(img_rows, img_cols, 1)
                test[sample_id][character_id] = 1
                sample_id += 1
            character_id += 1

    return train, test

def prepare_grouped_data_for_keras(data):
    grouped_data = [[[] for i in range(len(lang))] for lang in data]
     
    for l, language in enumerate(data):        
        for c, character in enumerate(language):
            for image in character:
                grouped_data[l][c].append(image.reshape(img_rows, img_cols, 1))            

    return grouped_data
