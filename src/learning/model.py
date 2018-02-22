'''
Created on January 12th, 2018
author: Julian Weisbord
sources: https://www.tensorflow.org/versions/r1.2/get_started/mnist/pros
         https://www.tensorflow.org/versions/r1.2/get_started/mnist/beginners
         https://www.youtube.com/watch?v=mynJtLhhcXk
description: Convolutional Neural Network that takes different images and
                classifies them into 5 different categories.

'''
import tensorflow as tf
from image_capture.prepare_data import PrepareData


# Definitions and Constants
CLASSES = ['bowl', 'calculator', 'cell_phone', 'notebook']
NUM_OBJECTS = 5  # Number of different objects per object class
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
COLOR_CHANNELS = 3
WEIGHT_SIZE = 5
BATCH_SIZE = 36
KEEP_RATE = 0.8
N_EPOCHS = 60
FC_NEURON_SIZE = 1024  # Chosen randomly
N_CLASSES = len(CLASSES)
FC_NUM_FEATURES = IMAGE_WIDTH * IMAGE_HEIGHT * N_CLASSES
TRAIN_PATH = '../image_data/captured_cropped'
VALIDATION_SIZE = .2
LEARNING_RATE = .001

WEIGHTS = {
    # Take 1 input, produce 32 output features, Convolution window is  WEIGHT_SIZE * WEIGHT_SIZE
    'W_conv1':tf.Variable(tf.random_normal([WEIGHT_SIZE, WEIGHT_SIZE, COLOR_CHANNELS, 32])),
    'W_conv2':tf.Variable(tf.random_normal([WEIGHT_SIZE, WEIGHT_SIZE, 32, 64])),
    'W_fc':tf.Variable(tf.random_normal([FC_NUM_FEATURES, FC_NEURON_SIZE])),
    'out':tf.Variable(tf.random_normal([FC_NEURON_SIZE, N_CLASSES]))}
BIASES = {
    'b_conv1':tf.Variable(tf.random_normal([32])),
    'b_conv2':tf.Variable(tf.random_normal([64])),
    'b_fc':tf.Variable(tf.random_normal([FC_NEURON_SIZE])),
    'out':tf.Variable(tf.random_normal([N_CLASSES]))}


def grab_dataset():
    '''
    Description: This function grabs the collected image data from prepare_data.py
    Return: <Tuple of Datasets> The training and validation datasets.
    '''
    print("Grabbing Data...")
    image_data = PrepareData()
    train_data, valid_data = image_data.read_train_sets(TRAIN_PATH, NUM_OBJECTS, CLASSES,
                                                        (IMAGE_WIDTH, IMAGE_HEIGHT),
                                                        VALIDATION_SIZE)
    return train_data, valid_data

def model_setup(x, keep_prob):
    '''
    Description: Create model layers and apply activations.
    Input: x <Tensor> is the input layer Tensorflow Placeholder,
               keep_prob <float> is a parameter used for dropout.
    Return: <Tensor> Softmax predicted class.
    '''

    x = tf.reshape(x, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS])
    conv1 = tf.nn.relu(conv2d(x, WEIGHTS['W_conv1']) + BIASES['b_conv1'])
    print("relu'd conv1 shape: ", conv1.shape)
    conv1 = maxpool2d(conv1)
    print("pool'd conv1 shape: ", conv1.shape)
    conv2 = tf.nn.relu(conv2d(conv1, WEIGHTS['W_conv2']) + BIASES['b_conv2'])
    print("relu'd conv2 shape: ", conv2.shape)
    conv2 = maxpool2d(conv2)
    print("pool'd conv2 shape: ", conv2.shape)
    # All of the neurons in conv2 will connect to every neuron in fc
    layer_shape = conv2.get_shape()
    print("Layer shape after convolution: ", layer_shape)

    num_features = layer_shape[1:4].num_elements()
    print("Layer shape 1 to 4", layer_shape[1:4])
    print("num_features after convolution", num_features)
    flatten = tf.reshape(conv2, [-1, num_features])  # shape is [batch_size, features], -1 makes dynamic batch_size
    # fc = tf.reshape(conv2, [-1, 7*7*64])  # shape is [batch_size, features], -1 makes dynamic batch_size
    fc = tf.nn.relu(tf.matmul(flatten, WEIGHTS['W_fc']) + BIASES['b_fc'])
    # fc = tf.nn.dropout(fc, keep_prob)
    print("fc shape: ", fc.get_shape())
    output = tf.matmul(fc, WEIGHTS['out']) + BIASES['out']
    return output

def loss(prediction, y):
    '''
    Description: Calculate loss function and perform Gradient Descent
    Input: prediction <Tensor> softmax prediction, y <Tensor> output layer Placeholder
    Return: optimizer <Tensor Operation> applies Adam Gradient Descent on the models,
                cost <Tensor> the cross_entropy loss function

    '''
    print("logits shape: {} labels shape: {}".format(prediction.shape, y.shape))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    return optimizer, cost

def train(x, y, keep_prob):
    '''
    Description: This function iteratively trains the model by applying training samples
                     and then updates the weights with Gradient Descent.
    Input: x <Tensor> is the input layer Placeholder, y <Tensor> output layer Placeholder,
               keep_prob <float> is a parameter used for dropout.
    Return: None
    '''
    prediction = model_setup(x, keep_prob)
    print("prediction shape: ", prediction.get_shape())
    optimizer, cost = loss(prediction, y)
    train_data, valid_data = grab_dataset()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(N_EPOCHS):
            epoch_loss = 0
            for _ in range(int(train_data.num_examples / BATCH_SIZE)):
                print("Num train data examples!!!!! ", train_data.num_examples)
                epoch_x, epoch_y = train_data.next_batch(BATCH_SIZE)
                # print("epoch_x Shape {}, epoch_y Shape {}".format(epoch_x.shape, epoch_y.shape))
                # _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: KEEP_RATE})
                # print("Part 2 logits shape: {} labels shape: {}".format(prediction.shape, y.shape))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                # print("Part 3 logits shape: {} labels shape: {}".format(prediction.shape, y.shape))
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', N_EPOCHS, 'loss:', epoch_loss)
        # Compute the accuracy of the model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:valid_data.images, y:valid_data.labels, keep_prob: 1.}))

def conv2d(x, W):
    '''
    Description: Convolves the input image with the weight matrix, one pixel at a time.
    Input: x <Tensor> is the input layer Placeholder.
    Return: <Tensor Operation> A 2d Convolution.
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    '''
    Description: Maxpooling is used to simplify and take the extreme values. It slides a
    2*2 window 2 pixels at a time.
    Input: x <Tensor> is the input layer Placeholder.
    Return: <Tensor Operation> Maxpooling operation with a 2*2 window size and 2*2 strides.
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main():
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS])
    keep_prob = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    train(x, y, keep_prob)


if __name__ == '__main__':
    main()
