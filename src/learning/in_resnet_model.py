'''
Created on March 4th, 2018
author: Julian Weisbord
sources: https://github.com/kwotsin/transfer_learning_tutorial
description:

'''

# External Imports
import time
import sys
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging


# Local Imports
from image_capture.prepare_data import PrepareData
from inception_resnet import inception_resnet_v2 as res

# Definitions and Constants
CLASSES = ['bowl', 'calculator', 'cell_phone', 'notebook']
NUM_OBJECTS = 5  # Number of different objects per object class
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
COLOR_CHANNELS = 3
BATCH_SIZE = 50
KEEP_RATE = 0.8
N_EPOCHS = 50
N_CLASSES = len(CLASSES)
DEFUALT_TRAIN_PATH = '../image_data/captured_cropped'
SAVED_MODEL_PATH = 'robot-environment-model'
VALIDATION_SIZE = .2
LEARNING_RATE = .0002
LEARNING_RATE_DECAY_FACTOR = 0.7
NUM_EPOCHS_BEFORE_DECAY = 2
# Create the file pattern of your TFRecord files so that it could be recognized later on
FILE_PATTERN = 'flowers_%s_*.tfrecord'
#install https://github.com/tensorflow/models/tree/master/research/slim
CHECKPOINT_FILE = './resnet_v2_152.ckpt'
LOG_DIR = './log'


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

# def make_tfrecord():
#     labels_to_name = {}
#     for line in labels:
#         label, string_name = line.split(':')
#         string_name = string_name[:-1] #Remove newline
#         labels_to_name[int(label)] = string_name
def retrain(x, train_path):
    train_data, valid_data = grab_dataset(train_path)
    # items_to_descriptions = {
    #     'image': '3-channel RGB picture of an object',
    #     'label': 'A label that is as such -- 0:bowl, 1:calculator, 2:cell_phone, 3:notebook'
    # }

    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        with tf.contrib.slim.arg_scope(res.inception_resnet_v2_arg_scope()):
            logits, end_points = res.inception_resnet_v2(train_data.images, num_classes=N_CLASSES, is_training=True)
        # Restoring the model variables from checkpoint file will result in
            # a number of classes mismatch unless these are excluded.
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)

        one_hot_labels = train_data.labels
        print("end_points", end_points)
        # tf.nn.softmax_cross_entropy_with_logits_v2
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(onehot_labels=one_hot_labels, logits=logits)
        total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well

        #Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()
        num_batches_per_epoch = int(train_data.num_examples /BATCH_SIZE)
        #Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate=LEARNING_RATE,
            global_step=global_step,
            decay_steps=int(NUM_EPOCHS_BEFORE_DECAY * num_batches_per_epoch),
            decay_rate=LEARNING_RATE_DECAY_FACTOR,
            staircase=True)
        #Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = tf.contrib.slim.learning.create_train_op(total_loss, optimizer)

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.metrics.accuracy(train_data.class_name,
                                                                          predictions)
        metrics_op = tf.group(accuracy_update, probabilities)
        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        #Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, CHECKPOINT_FILE)

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=LOG_DIR, summary_op=None, init_fn=restore_fn)


        #Run the managed session
        with sv.managed_session() as sess:
            for step in xrange(num_batches_per_epoch * N_EPOCHS):
                #At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, N_EPOCHS)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits,
                                                                                                   probabilities,
                                                                                                   predictions,
                                                                                                   train_data.class_name])
                    print 'logits: \n', logits_value
                    print 'Probabilities: \n', probabilities_value
                    print 'predictions: \n', predictions_value
                    print 'Labels:\n:', labels_value

                #Log the summaries every 10 step.
                if step % 10 == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step, metrics_op)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                #If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step, metrics_op)

            #We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            # saver.save(sess, "./flowers_model.ckpt")
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)




def train_step(sess, train_op, global_step, metrics_op):
    '''
    Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
    '''
    #Check the time for each sess run
    start_time = time.time()
    total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
    time_elapsed = time.time() - start_time

    #Run the logging to print some results
    logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

    return total_loss, global_step_count


def main():
    if len(sys.argv) != 2:
        print("Using default training dataset path")
        train_path = DEFUALT_TRAIN_PATH
    else:
        train_path = sys.argv[1]

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS])
    # keep_prob = tf.placeholder(tf.float32)
    #y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    retrain(x, train_path)


if __name__ == '__main__':
    main()
