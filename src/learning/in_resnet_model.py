'''
Created on March 4th, 2018
author: Julian Weisbord
sources:
description:

'''

# External Imports
import time
import sys
import tensorflow as tf
# Local Imports
from image_capture.prepare_data import PrepareData

# Definitions and Constants
CLASSES = ['bowl', 'calculator', 'cell_phone', 'notebook']
NUM_OBJECTS = 5  # Number of different objects per object class
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
COLOR_CHANNELS = 3
BATCH_SIZE = 50
KEEP_RATE = 0.8
N_EPOCHS = 800
N_CLASSES = len(CLASSES)
DEFUALT_TRAIN_PATH = '../image_data/captured_cropped'
SAVED_MODEL_PATH = 'robot-environment-model'
VALIDATION_SIZE = .2
LEARNING_RATE = .001


def grab_dataset(train_path):
    '''
    Description: This function grabs the collected image data from prepare_data.py
    Return: <Tuple of Datasets> The training and validation datasets.
    '''

    image_data = PrepareData()
    train_data, valid_data = image_data.read_train_sets(train_path, NUM_OBJECTS, CLASSES,
                                                        (IMAGE_WIDTH, IMAGE_HEIGHT),
                                                        VALIDATION_SIZE)
    return train_data, valid_data

def retrain(x):

    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(x, num_classes=1001,is_training=False)
    predictions = end_points["PreLogitsFlatten"] #you can try with other layers too. Intermediate layers


def main():
    if len(sys.argv) != 2:
        print("Using default training dataset path")
        train_path = DEFUALT_TRAIN_PATH
    else:
        train_path = sys.argv[1]

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS])
    keep_prob = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    retrain(x, y, keep_prob, train_path)


if __name__ == '__main__':
    main()
