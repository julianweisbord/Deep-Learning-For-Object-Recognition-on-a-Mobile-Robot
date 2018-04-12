'''
Created on March 14th, 2018
author: Julian Weisbord
sources: https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb
         http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
description:

'''
# External Imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
import numpy as np
import tensorflow as tf

# Local Imports
import model as mdl
from image_capture.novel_image import PrepareNovelData


SAVED_MODEL_PATH = 'robot-environment-model/.data-00000-of-00001'
NEW_IMAGE_PATH = '../image_data/cropped_new_task/'
CLASSES = ['book', 'chair', 'mug', 'screwdriver', 'stapler']
N_CLASSES = len(CLASSES)
NUM_OBJECTS_PER_CLASS = 1
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
COLOR_CHANNELS = 3
VALIDATION_SIZE = .2
LAMBDA_OPTIONS = [0, 1, 2, 5, 15]
LEARNING_RATE = .001
ITERATIONS = 200

def grab_dataset(self, train_path):
    '''
    Description: This function grabs the collected image data from prepare_data.py
    Return: <Tuple of Datasets> The training and validation datasets.
    '''

    image_data = PrepareNovelData()
    # PrepareNovelData should have a check to see how many objects are actually in train_path
        # and only use those since the number may be different than the constant "CLASSES"
        # Also notice NUM_OBJECTS_PER_CLASS is not in read_data_sets_facade, this may be
        # different each type so it should not be a constant. We will have to determine
        # this value from the directory structure.
    train_data, valid_data = image_data.read_data_sets_facade(train_path, CLASSES,
                                                              (IMAGE_WIDTH, IMAGE_HEIGHT),
                                                              VALIDATION_SIZE)
    return train_data, valid_data


class EWCModel:
    def __init__(self, new_train_path, model, model_type):
        self.accuracy = 0
        self.F_accum = []
        self.var_list = []
        self.optimal_vars = []
        self.model = model
        self.model_type = model_type
        self.ewc_loss = 0

        self.x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS], name = "x")
        self.keep_prob = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32, shape=[None, N_CLASSES], name="y")

        self.create_vars()
        # Restore optimal vars for model task
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("w1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        b1 = graph.get_tensor_by_name("b1:0")
        b2 = graph.get_tensor_by_name("b1:0")
        self.optimal_vars.append(w1, b1, w2, b2)


    def create_vars(self):
        w1 = tf.Variable(tf.random_normal([WEIGHT_SIZE, WEIGHT_SIZE, COLOR_CHANNELS, 32]))
        b1 = tf.Variable(tf.random_normal([32]))
        w2 = tf.Variable(tf.random_normal([WEIGHT_SIZE, WEIGHT_SIZE, 32, 64]))
        b2 = tf.Variable(tf.random_normal([64])
        self.var_list = [w1, b1, w2, b2]

    def new_task_model(self):
        if self.model_type == "model":
            prediction = mdl.model_setup(self.x, self.keep_prob)
            with tf.name_scope('SequentialLearningCrossEntropy'):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=self.y))
            self.ewc_loss = cost

        elif self.model_type == "in_resnet":
            pass

    def update_opt_weights(self):
        pass

    def compute_fisher(self):
        pass

    def update_ewc_loss(self, lam):
        for i in range(len(self.var_list)):
            self.ewc_loss += lam/2 * tf.reduce_sum(tf.multiply(self.F_accum[i].astype(np.float32),
                                                           tf.square(self.var_list[i] - self.optimal_vars[i])))
            # Probably should switch to adam optimizer
            self.train_step = tf.train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.ewc_loss)

    def save_optimal_vars(self):
        pass


def main():
    # Determines which command line args and default args to use.
    try:
        if len(sys.argv) < 2:
            print("Using default training dataset path")
            new_train_path = NEW_IMAGE_PATH
            model_type = "model"
        else:
            if sys.argv[1]:
                new_train_path = sys.argv[1]
            else:
                new_train_path = NEW_IMAGE_PATH
            model_type = sys.argv[2]
    except:
        print("The correct arguments are: 'sequential_learning.py new_train_path<None|string> model_type<in_resnet|model>'")


    with tf.Session() as sess:
        try:
            tf.Saver.restore(sess, SAVED_MODEL_PATH)
        except:
            print("You need to have already run model.py without sequential learning")
        online = EWCModel(new_train_path, sess, model_type)
        online.new_task_model()
        online.update_loss()
        online.update_ewc_loss(LAMBDA_OPTIONS[4])

        train_data, valid_data = grab_dataset(new_train_path)

        for iteration in range(NUM_ITERATIONS):
            epoch_x, epoch_y = train_data.next_batch(BATCH_SIZE)

            sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            online.train_step.run(feed_dict={x: epoch_x, y_: epoch_y})

if __name__ == '__main__':
    main()
