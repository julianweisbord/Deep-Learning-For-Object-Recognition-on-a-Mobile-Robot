'''
Created on March 14th, 2018
author: Julian Weisbord
sources: https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb
         http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
description:

'''
# External Imports
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
import numpy as np
import tensorflow as tf

# Local Imports
import model as mdl
from data_prep.prepare_data import PrepareData


SAVED_MODEL_PATH = 'robot-environment-model/.data-00000-of-00001'
NEW_IMAGE_PATH = '../image_data/cropped_new_task/'
SAVED_MODEL_BASE = './robot-environment-model/'
SAVED_MODEL_PATH = 'robot-environment-model/model.meta'
CLASSES = ['book', 'chair', 'mug', 'screwdriver', 'stapler']
N_CLASSES = len(CLASSES)
NUM_OBJECTS_PER_CLASS = 1
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
WEIGHT_SIZE = 5
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
COLOR_CHANNELS = 3
VALIDATION_SIZE = .2
LAMBDA_OPTIONS = [0, 1, 2, 5, 15]
LEARNING_RATE = .001
COMPARE_TO_SGD = True
NUM_ITERATIONS = 10
BATCH_SIZE = 10

def grab_dataset(train_path):
    '''
    Description: This function grabs the collected image data from prepare_data.py
    Return: <Tuple of Datasets> The training and validation datasets.
    '''

    image_data = PrepareData()
    # PrepareNovelData should have a check to see how many objects are actually in train_path
        # and only use those since the number may be different than the constant "CLASSES"
        # Also notice NUM_OBJECTS_PER_CLASS is not in read_data_sets_facade, this may be
        # different each type so it should not be a constant. We will have to determine
        # this value from the directory structure.
    train_data, val_data = image_data.read_train_sets(train_path,
                                                      (IMAGE_WIDTH, IMAGE_HEIGHT),
                                                      VALIDATION_SIZE)
    return train_data, val_data


class EWCModel():
    def __init__(self, new_train_path, model, model_type, x, y_):
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
        b2 = graph.get_tensor_by_name("b2:0")
        self.optimal_vars = [w1, b1, w2, b2]


    def create_vars(self):
        w1 = tf.Variable(tf.random_normal([WEIGHT_SIZE, WEIGHT_SIZE, COLOR_CHANNELS, 32]))
        b1 = tf.Variable(tf.constant(0.1, shape=[32]))
        w2 = tf.Variable(tf.random_normal([WEIGHT_SIZE, WEIGHT_SIZE, 32, 64]))
        b2 = tf.Variable(tf.constant(0.1, shape=[64]))
        self.var_list = [w1, b1, w2, b2]

    def new_task_model(self):
        # Set up the second task's model
        if self.model_type == "model":
            prediction = mdl.model_setup(self.x, self.keep_prob)
            with tf.name_scope('SequentialLearningCrossEntropy'):
                self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=self.y))
            self.vanilla_sgd()
            self.ewc_loss = self.cross_entropy

        elif self.model_type == "in_resnet":
            pass
        return prediction

    def update_opt_weights(self):
        pass

    def reassign_opt_weights(self):
        # Replace original task's weights with sequential learning weights
        pass

    def vanilla_sgd(self):
        with tf.name_scope('StochasticGradientDescent'):
            self.train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.cross_entropy)


    def compute_fisher(self, prediction, imgset, sess, num_samples=200, plot=False):
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))
        print("F.accum: ", self.F_accum)

        class_ind = tf.to_int32(tf.multinomial(tf.log(prediction), 1)[0][0])

        for img_num in range(num_samples):
            # rand_image = np.random.randint(imgset.shape[0])
            # For each image, calculate derivatives of sum of log(prediction) of each image
                # w.r.t the weights of the new task
            ders = sess.run(tf.gradients(tf.log(prediction[0,class_ind]), self.var_list),
                            feed_dict={self.x: imgset[img_num]})
            # Square ders and add them to total
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])

        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples

    def update_ewc_loss(self, lam):
        for i in range(len(self.var_list)):
            self.ewc_loss += lam/2 * tf.reduce_sum(tf.multiply(self.F_accum[i].astype(np.float32),
                                                               tf.square(self.var_list[i] - self.optimal_vars[i])))
            # Probably should switch to adam optimizer
            self.train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.ewc_loss)

    def train(self, online, sess, display_freq, train_data, val_data,x, y_, lams, ewc=False):
        if ewc:
            online.update_ewc_loss(lams)
        else:
            online.vanilla_sgd()

        for _ in range(NUM_ITERATIONS):
            epoch_x, epoch_y = train_data.next_batch(BATCH_SIZE)

            online.train_step.run(feed_dict={x: epoch_x, y_: epoch_y})

    def save_optimal_vars(self):
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def plot_test_acc(self, plot_handles):
        plt.legend(handles=plot_handles, loc="center right")
        plt.xlabel("Iterations")
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1)
        display.display(plt.gcf())
        display.clear_output(wait=True)


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
            saved_model = tf.train.import_meta_graph(SAVED_MODEL_PATH)
            saved_model.restore(sess, tf.train.latest_checkpoint(SAVED_MODEL_BASE))
        except:
            print("You need to have already run model.py without sequential learning")
        display_freq = 10
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS], name = "x")
        y_ = tf.placeholder(tf.float32, shape=[None, N_CLASSES], name="y")
        train_data, val_data = grab_dataset(new_train_path)
        online = EWCModel(new_train_path, sess, model_type, x, y_)

        prediction = online.new_task_model()
        online.compute_fisher(prediction, val_data.images, sess, num_samples=len(train_data.images[0]))
        online.train(online, sess, display_freq, train_data, val_data, x, y_, LAMBDA_OPTIONS[4], ewc=True)

        online.save_optimal_vars()

if __name__ == '__main__':
    main()
