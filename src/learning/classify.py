'''
Created on March 2nd, 2018
author: Julian Weisbord
description: This is the inference layer, it runs a novel image set through
                the model and outputs the model's accuracy.

'''

# External Imports
import sys
import numpy as np
import tensorflow as tf
# Local Imports
from image_capture.novel_image_prep import PrepareNovelData

# Definitions and Constants
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
COLOR_CHANNELS = 3
SAVED_MODEL_PATH = 'robot-environment-model'

def grab_dataset(dataset_path):
    '''
    Description: This function grabs the collected image data from novel_image_prep.py
    Return: <Tuple of Datasets> The training and validation datasets.
    '''

    pred_data = PrepareNovelData(dataset_path)
    return pred_data

def classify(dataset):
    x_batch = tf.reshape(dataset, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS])
    with tf.Session() as sess:
        saved_model = tf.train.import_meta_graph(SAVED_MODEL_PATH)
        saved_model.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        y_pred = graph.get_tensor_by_name("prediction:0")

        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y:0")
        y_test_images = np.zeros((1, 2))
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        prediction = sess.run(y_pred, feed_dict=feed_dict_testing)

    return prediction

def main():
    if len(sys.argv) != 2:
        print("Please provide the path to your dataset")
        exit()
    dataset = grab_dataset(sys.argv[1])
    prediction = classify(dataset)
    print("Prediction: ", prediction)


if __name__ == '__main__':
    main()
