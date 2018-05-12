'''
Created on March 2nd, 2018
author: Julian Weisbord
description: This is the inference layer, it runs a novel image set through
                the model and outputs the model's accuracy. If the novel
                image set is accurate enough, this modules adds it to
                the full data set.

'''

# External Imports
import sys
import numpy as np
import tensorflow as tf
# Local Imports
from image_capture.prepare_data import PrepareData

# Definitions and Constants
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
COLOR_CHANNELS = 3
SAVED_MODEL_PATH = 'robot-environment-model/model.meta'
DEFAULT_DATA_PATH = '../image_data/cropped_inference'

def grab_dataset(dataset_path):
    '''
    Description: This function grabs the collected image data from novel_image_prep.py
    Return: <Tuple of Datasets> The training and validation datasets.
    '''
    pred_data = PrepareData()
    dataset = pred_data.read_train_sets(dataset_path,
                                             (IMAGE_WIDTH, IMAGE_HEIGHT),
                                             validation_size=0, classes=None, num_objects=None)
    return dataset

def classify(dataset):
    # print("dataset", dataset.images)
    # x_batch, labels = tf.reshape(dataset, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS])

    batch = dataset.images
    print("batch shape", batch.shape)

    # batch.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS)
    # exit()
    labels = dataset.labels
    y = tf.placeholder(tf.float32, shape=[None, len(labels)], name="y")
    with tf.Session() as sess:

        saved_model = tf.train.import_meta_graph(SAVED_MODEL_PATH)
        saved_model.restore(sess, tf.train.latest_checkpoint('./robot-environment-model/'))

        graph = tf.get_default_graph()
        # for op in graph.get_operations():
        #     print(str(op.values()))
        # print(tf.all_variables())
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y:0")
        prediction = graph.get_tensor_by_name("Prediction/add:0")

        # prediction =
        # exit()
        # y_pred = graph.get_tensor_by_name("output:0")
        # y_test_images = np.zeros((1, 2))
        print("image shape: ", batch.shape)
        print("label shape: {}".format(labels.shape))
        print("y_true.shape: ", y_true.shape)
        feed_dict_testing = {x: batch, y_true: labels}
        result = sess.run(prediction, feed_dict=feed_dict_testing)

    return result

def main():
    if len(sys.argv) == 2:
        prediction_data = sys.argv[1]

    else:
        print("Using default dataset path instead of command line args")
        prediction_data = DEFAULT_DATA_PATH

    dataset = grab_dataset(prediction_data)
    prediction = classify(dataset)
    print("Prediction: ", prediction)


if __name__ == '__main__':
    main()
