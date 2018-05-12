'''
date: 2/7/2018
author: Miles McCall
sources: 
description: Take a new set of images as input and prepares them as a tensorflow data set"
'''

# External Imports
import time
import sys
	#import tensorflow as tf
# Local Imports
from prepare_novel_data import PrepareData

'''
# Definitions and Constants
CLASSES = ['bowl', 'calculator', 'cell_phone', 'notebook']
NUM_OBJECTS = 5  # Number of different objects per object class
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
COLOR_CHANNELS = 3
WEIGHT_SIZE = 5
BATCH_SIZE = 50
KEEP_RATE = 0.8
N_EPOCHS = 800
FC_NEURON_SIZE = 1024  # Chosen randomly for now
N_CLASSES = len(CLASSES)
FC_NUM_FEATURES = IMAGE_WIDTH * IMAGE_HEIGHT * N_CLASSES
DEFUALT_TRAIN_PATH = '../image_data/captured_cropped'
VALIDATION_SIZE = .2
LEARNING_RATE = .001
'''

# Definitions and Constants
NUM_OBJECTS = 1  # Number of different objects per object class
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
N_CLASSES = 1
DEFUALT_TRAIN_PATH = '../image_data/captured_cropped'
VALIDATION_SIZE = .2

def append_novel_images(obj_class, img_path):
    image_data = PrepareData()

    train_data, valid_data = image_data.read_train_sets(
	  img_path, 
	  NUM_OBJECTS, 
	  obj_class, 
	  (IMAGE_WIDTH, IMAGE_HEIGHT), 
	  VALIDATION_SIZE
	  )

    return train_data, valid_data

def main():
    obj_class = sys.argv[1]
    img_path  = sys.argv[2]
    train_data, valid_data = append_novel_images(obj_class, img_path)

    print train_data.img_names
    print "~~~"
    print valid_data.img_names

if __name__ == '__main__':
    main()
