'''
Created on January 12th, 2018
author: Julian Weisbord
sources: https://www.tensorflow.org/versions/r1.2/get_started/mnist/pros
         https://www.tensorflow.org/versions/r1.2/get_started/mnist/beginners
         https://www.youtube.com/watch?v=fBVEXKp4DIc
         https://www.youtube.com/watch?v=mynJtLhhcXk
description: Convolutional Neural Network that takes different images and
                classifies them into 5 different categories.

'''

# External Imports
import time
import sys
import numpy as np
import tensorflow as tf
# Local Imports
from image_capture.prepare_data import PrepareData

# Definitions and Constants
CLASSES = ['book', 'chair', 'mug', 'screwdriver', 'stapler']
NUM_OBJECTS_PER_CLASS = 5  # Number of different objects per object class
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
COLOR_CHANNELS = 3
WEIGHT_SIZE = 5
BATCH_SIZE = 50  # Number of images per step or iteration
KEEP_RATE = 0.55
N_EPOCHS = 3500  # One iteration over all of the training data
FC_NEURON_SIZE = 1024  # Chosen randomly for now
N_CLASSES = len(CLASSES)
FC_NUM_FEATURES = np.int32(IMAGE_WIDTH * IMAGE_HEIGHT * N_CLASSES * .8)
DEFAULT_TRAIN_PATH = '../image_data/cropped'
SAVED_MODEL_PATH = 'robot-environment-model/model'
LOGDIR = "../vis/tfviz"
VALIDATION_SIZE = .2
LEARNING_RATE = .00102


WEIGHTS = {
    # W_conv1 : Take 1 input, produce 32 output features, Convolution window is  WEIGHT_SIZE * WEIGHT_SIZE
    'W_conv1':tf.Variable(tf.random_normal([WEIGHT_SIZE, WEIGHT_SIZE, COLOR_CHANNELS, 32]), name='w1'),
    'W_conv2':tf.Variable(tf.random_normal([WEIGHT_SIZE, WEIGHT_SIZE, 32, 64]), name='w2'),
    'W_fc':tf.Variable(tf.random_normal([FC_NUM_FEATURES, FC_NEURON_SIZE]), name='w_fc'),
    'out':tf.Variable(tf.random_normal([FC_NEURON_SIZE, N_CLASSES]), name='w_softmax')}
BIASES = {
    # 'b_conv1':tf.Variable(tf.random_normal([32]), name='b1'),
    # 'b_conv2':tf.Variable(tf.random_normal([64]), name='b2'),
    # 'b_fc':tf.Variable(tf.random_normal([FC_NEURON_SIZE]), name='b_fc'),
    # 'out':tf.Variable(tf.random_normal([N_CLASSES]), name='b_softmax')
    'b_conv1':tf.Variable(tf.constant(0.1, shape=[32]), name='b1'),
    'b_conv2':tf.Variable(tf.constant(0.1, shape=[64]), name='b2'),
    'b_fc':tf.Variable(tf.constant(0.1, shape=[FC_NEURON_SIZE]), name='b_fc'),
    'out':tf.Variable(tf.constant(0.1, shape=[N_CLASSES]), name='b_softmax')}


def grab_dataset(train_path):
    '''
    Description: This function grabs the collected image data from prepare_data.py
    Return: <Tuple of Datasets> The training and validation datasets.
    '''

    image_data = PrepareData()
    train_data, valid_data = image_data.read_train_sets(train_path,
                                                        (IMAGE_WIDTH, IMAGE_HEIGHT),
                                                        VALIDATION_SIZE, CLASSES,
                                                        NUM_OBJECTS_PER_CLASS)
    return train_data, valid_data

def model_setup(x, keep_prob):
    '''
    Description: Create model layers and apply activations.
    Input: x <Tensor> is the input layer Tensorflow Placeholder,
               keep_prob <float> is a parameter used for dropout.
    Return: <Tensor> Softmax predicted class.
    '''

    x = tf.reshape(x, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS])
    tf.summary.image('input', x, 3)
    conv1 = conv2d(x, WEIGHTS['W_conv1'], BIASES['b_conv1'], "Convolution1")
    conv2 = conv2d(conv1, WEIGHTS['W_conv2'], BIASES['b_conv2'], "Convolution2")
    conv2shape = conv2.get_shape().as_list()
    print("conv2shape: ", conv2shape)
    # All of the neurons in conv2 will connect to every neuron in fc
    flatten = tf.reshape(conv2, [-1, conv2shape[1] * conv2shape[2] * conv2shape[3]])
    print("flatten shape", flatten.get_shape())
    # with tf.name_scope('FullyConnected'):
    #     fc = tf.nn.relu(tf.matmul(flatten, WEIGHTS['W_fc']) + BIASES['b_fc'])
    fc1 = fc_layer(flatten, WEIGHTS['W_fc'], BIASES['b_fc'], "FullyConnected")
    # Apply Dropout
    with tf.name_scope('Dropout'):
        fc1 = tf.nn.dropout(fc1, keep_prob)
    # Final fully connected layer but without RELU
    with tf.name_scope('Prediction'):
        # output = tf.Variable("softmax", shape=[None, 5], initializer=tf.zeros_initializer)
        output = tf.matmul(fc1, WEIGHTS['out']) + BIASES['out']
        print ("output", output)
        print("output shape:", output.shape)
        return output

def loss(prediction, y):
    '''
    Description: Calculate loss function and perform Gradient Descent
    Input: prediction <Tensor> softmax prediction, y <Tensor> output layer Placeholder
    Return: optimizer <Tensor Operation> applies Adam Gradient Descent on the models,
                cost <Tensor> the cross_entropy loss function

    '''
    with tf.name_scope('CrossEntropy'):
        y_pred = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)
        cost = tf.reduce_mean(y_pred)
        tf.summary.scalar("CrossEntropy", cost)
    with tf.name_scope('Train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    return optimizer, cost

def train(x, y, keep_prob, train_path, saved_model_path=False, viz_name=False):
    '''
    Description: This function iteratively trains the model by applying training samples
                     and then updates the weights with Gradient Descent.
    Input: x <Tensor> is the input layer Placeholder, y <Tensor> output layer Placeholder,
               keep_prob <float> is a parameter used for dropout.
    Return: None
    '''
    start_time = time.time()

    prediction = model_setup(x, keep_prob)
    optimizer, cost = loss(prediction, y)
    train_data, valid_data = grab_dataset(train_path)
    save_model = tf.train.Saver()

    # Compute the accuracy of the model
    with tf.name_scope('Accuracy'):
        # Returns index with largest value across axes of a Tensor
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        tf.summary.scalar("Accuracy", accuracy)
    # Write all summaries just once
    summ = tf.summary.merge_all()
    if viz_name:
        writer = tf.summary.FileWriter("../vis/" + viz_name)
    else:
        writer = tf.summary.FileWriter(LOGDIR)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(N_EPOCHS):
            epoch_x, epoch_y = train_data.next_batch(BATCH_SIZE)
            train_accuracy, s = sess.run([accuracy, summ], feed_dict={x:epoch_x, y:epoch_y, keep_prob: KEEP_RATE})
            # correct_val = sess.run([correct], feed_dict={x:epoch_x, y:epoch_y, keep_prob: KEEP_RATE})
            if epoch % 10 == 0:
                writer.add_summary(s, epoch)
            print('step %d, training accuracy %g' % (epoch, train_accuracy))
            sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: KEEP_RATE})
            print('Test Accuracy:', accuracy.eval({x:valid_data.images, y:valid_data.labels, keep_prob: 1.0}))

        if not saved_model_path:
            saved_model_path = SAVED_MODEL_PATH
        save_model.save(sess, saved_model_path)
        writer.add_graph(sess.graph)
    end_time = time.time() - start_time
    print("Total time", end_time)

def conv2d(inp, W, b, name):
    '''
    Description: Convolves the input image with the weight matrix, one pixel at a time.
    Input: input <Tensor> is the input layer Placeholder.
    Return: <Tensor Operation> A 2d Convolution.

    '''
    with tf.name_scope(name):
        conv = tf.nn.conv2d(inp, W, strides=[1, 1, 1, 1], padding='SAME') + b
        conv_relu = tf.nn.relu(conv)
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Biases", b)
        tf.summary.histogram("RELU Activation", conv_relu)
        conv_pooled = maxpool2d(conv_relu)
        return conv_pooled

def fc_layer(inp, W, b, name):
    '''
    Description: Convolves the input image with the weight matrix, one pixel at a time.
    Input: input <Tensor> is the input layer Placeholder.
    Return: <Tensor Operation> A 2d Convolution.

    '''
    with tf.name_scope(name):
        fc_relu = tf.nn.relu(tf.matmul(inp, W) + BIASES['b_fc'])
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Biases", b)
        tf.summary.histogram("RELU Activation", fc_relu)
        return fc_relu

def maxpool2d(x):
    '''
    Description: Maxpooling is used to simplify and take the extreme values. It slides a
    2*2 window 2 pixels at a time.
    Input: x <Tensor> is the input layer Placeholder.
    Return: <Tensor Operation> Maxpooling operation with a 2*2 window size and 2*2 strides.
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main():
    if len(sys.argv) != 2:
        print("Using default training dataset path")
        train_path = DEFAULT_TRAIN_PATH
    else:
        train_path = sys.argv[1]
    # None here allows us to pass batches of any number of images
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS], name = "x")
    keep_prob = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES], name="y")
    print("y: ", y)

    train(x, y, keep_prob, train_path)


if __name__ == '__main__':
    main()
