'''
Created on May 9th, 2018
author: Julian Weisbord
description: This is the inference layer, it runs a novel image set through
                the model and outputs the model's accuracy. If the novel
                image set is accurate enough, this modules adds it to
                the full data set.

'''
# External Imports
import os
import sys
import glob
import numpy as np
from keras.models import model_from_yaml
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
# Local Imports
from image_capture.prepare_data import PrepareData

# Definitions and Constants
IMAGE_HEIGHT = 139
IMAGE_WIDTH = 139
COLOR_CHANNELS = 3
SAVED_MODEL_PATH = 'robot-environment-model/in_resnet.yaml'
DEFAULT_DATA_PATH = '../image_data/cropped_inference'
CLASSES = ['book', 'chair', 'mug', 'screwdriver', 'stapler']
N_CLASSES = len(CLASSES)

def grab_dataset(dataset_path, classes=None, num_objects=0):
    '''
    # Description: This function grabs the collected image data from novel_image_prep.py
    # Return: <Tuple of Datasets> The training and validation datasets.
    # '''
    labels = []
    file_paths = []

    if not classes:
        classes = []
        for cls in os.listdir(DEFAULT_DATA_PATH):
            classes.append(cls)

    for field in classes:
        # Determine number of objects that were captured per object class
        print("classes: ", classes)
        if not num_objects:
            num_objects = 0
            for _ in os.listdir(DEFAULT_DATA_PATH + '/' + field):
                num_objects += 1

        print("num objects", num_objects)
        index = classes.index(field)
        print('Now going to read {} files (Index: {})'.format(field, index))

        for i in range(1, num_objects + 1):
            img = field + '_' + str(i) + '/images/'
            path = os.path.join(DEFAULT_DATA_PATH, field, img)
            files = glob.glob(path + '*')
            print("path", path)
            for img in files:
                label = np.zeros(N_CLASSES)
                label[index] = 1.0
                labels.append(label)
            file_paths.append(files)
    return file_paths, labels

def classify(file_paths, labels):
    # load YAML and restore model
    yaml_file = open(SAVED_MODEL_PATH, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)

    predictions = []
    count = 0
    for field_arr in file_paths:
        for img_path in field_arr:
            img = image.load_img(img_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
            img = image.img_to_array(img)
            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)
            preds = loaded_model.predict(x, batch_size=10, verbose=1)
            # cls = np.argmax(preds)
            print("Prediction: ", preds)
            print("count: ", count)
            print("label for this val: ", labels[count])
            count += 1
    return predictions, labels

def main():
    if len(sys.argv) == 2:
        prediction_data = sys.argv[1]
    else:
        print("Using default dataset path instead of command line args")
        prediction_data = DEFAULT_DATA_PATH

    file_paths, labels = grab_dataset(prediction_data)
    predictions, correct_values = classify(file_paths, labels)
    # print("Prediction: ", prediction)

if __name__ == '__main__':
    main()
