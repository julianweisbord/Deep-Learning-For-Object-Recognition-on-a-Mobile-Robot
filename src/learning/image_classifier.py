'''
Created on January 12th, 2018
author: Julian Weisbord
sources:
description: Convolutional Neural Network Model
'''
from image_capture.prepare_data import PrepareData
import tensorflow as tf

# Constants
# CLASSES = ['coffee_mug', 'book', 'chair', 'screwdriver', 'stapler']
CLASSES = ['bowl', 'calculator', 'cell_phone', 'notebook']
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_SIZE = 784  # 28 * 28 pixels, might change this
COLOR_CHANNELS = 3
WEIGHT_SIZE = 5
BATCH_SIZE = 128
KEEP_RATE = 0.8
N_EPOCHS = 15
FC_NEURON_SIZE = 1024  # Chosen randomly
N_CLASSES = len(CLASSES)
TRAIN_PATH = '../image_data/captured_cropped'
VALIDATION_SIZE = .2

WEIGHTS = {
    'W_conv1':tf.Variable(tf.random_normal([WEIGHT_SIZE, WEIGHT_SIZE, COLOR_CHANNELS, 32])),  # Convolve 5 * 5, take 1 input, produce 32 output features
    'W_conv2':tf.Variable(tf.random_normal([WEIGHT_SIZE, WEIGHT_SIZE, 32, 64])),
    'W_fc':tf.Variable(tf.random_normal([7*7*64, FC_NEURON_SIZE])),
    'out':tf.Variable(tf.random_normal([FC_NEURON_SIZE, N_CLASSES]))}
BIASES = {
    'b_conv1':tf.Variable(tf.random_normal([32])),
    'b_conv2':tf.Variable(tf.random_normal([64])),
    'b_fc':tf.Variable(tf.random_normal([FC_NEURON_SIZE])),
    'out':tf.Variable(tf.random_normal([N_CLASSES]))}


def grab_dataset():
    print("Grabbing Data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    image_data = PrepareData()
    train_data, valid_data = image_data.read_train_sets(TRAIN_PATH, CLASSES, VALIDATION_SIZE)
    print("Done Grabbing Data!!!!!!!!!!!!!11111111!!!!!!!!")
    return train_data, valid_data


def inference_model(x, keep_prob):
    '''
    Create model for inference
    '''
    train_data, valid_data = grab_dataset()

    x = tf.reshape(x, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS])  # Reshape to a 28 *28 tensor
    conv1 = tf.nn.relu(conv2d(x, WEIGHTS['W_conv1']) + BIASES['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = tf.nn.relu(conv2d(conv1, WEIGHTS['W_conv2']) + BIASES['b_conv2'])
    conv2 = maxpool2d(conv2)
    # All of the neurons in conv2 will connect to every neuron in fc
    fc = tf.reshape(conv2, [-1, 7*7*64])  # shape is [batch_size, features], -1 makes dynamic batch_size
    fc = tf.nn.relu(tf.matmul(fc, WEIGHTS['W_fc']) + BIASES['b_fc'])
    # fc = tf.nn.dropout(fc, keep_prob)
    output = tf.matmul(fc, WEIGHTS['out']) + BIASES['out']
    return output

def loss(prediction, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(cost)
    return optimizer, cost

def train(x, y, keep_prob):
    prediction = inference_model(x, keep_prob)
    optimizer,cost = loss(prediction, y)

    with tf.Session() as sess:
        exit()
        sess.run(tf.global_variables_initializer())
        for epoch in range(N_EPOCHS):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / BATCH_SIZE)):
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
                # _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: KEEP_RATE})
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', N_EPOCHS, 'loss:', epoch_loss)
        # Compute the accuracy of the model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels, keep_prob: 1.}))

def conv2d(x, W):
    '''
    Convolves the input image with the weight matrix, one pixel at a time.
    '''
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    '''
    Maxpooling is used to simplify and take the extreme values. It slides a
    2*2 window 2 pixels at a time.
    '''
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def main():
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
    keep_prob = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    train(x, y, keep_prob)


if __name__ == '__main__':
    main()
