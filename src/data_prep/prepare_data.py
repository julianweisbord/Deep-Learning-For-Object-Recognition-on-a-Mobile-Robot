'''
Created on February 1st, 2018
author: Julian Weisbord
sources: https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier
description: Module to prepare image data for tensorflow model.
'''

# External Imports
import os
import glob
import numpy as np
from sklearn.utils import shuffle
import cv2

# TODO: Caching Plan:
# if data is already in a file (first line of label_file is [last_data_change_time, last time file was written to])
    # if last data change made > last time label_file was written to
    #     go through all of the image data and write to label_file with new data changes
    # else
    #     use label_file for training
# else
    # train network by parsing images in folders
    # go through all of the image data and write to label_file with new data changes


class PrepareData():
    '''
    Description: This class takes a directory of images and converts them into
                    a numpy matrix of pixels and data labels. This class creates
                    datasets and makes them accessible to the model.
    '''

    def __init__(self):
        self.train = None
        self.valid = None

    def read_train_sets(self, train_path, image_size, validation_size=.2, classes=None, num_objects=None):
        '''
        Description: Helper function to load image data and create validation and training datasets.
        Input: train_path <string> the path to overacrching folder containing all image files,
               classes <list of strings> contains the names of the different image categories,
               image_size <tuple of ints> is the desired width and height of the input image,
               validation_size <float> is the percent/100 value for what portion of images
                   are used for validation
        Return: data_sets <tuple of Dataset() objects> are the training and validation datasets
        '''

        images, labels, img_names, class_name = self.load_train(train_path, image_size, classes, num_objects)
        images, labels, img_names, class_name = shuffle(images, labels, img_names, class_name)  # Randomize arrays

        if isinstance(validation_size, float):
            validation_size = int(validation_size * images.shape[0])
        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]
        validation_img_names = img_names[:validation_size]
        validation_class_name = class_name[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]
        train_img_names = img_names[validation_size:]
        train_class_name = class_name[validation_size:]

        if validation_size == 0:
            data_sets = Dataset(train_images, train_labels, train_img_names, train_class_name)
        else:

            self.train = Dataset(train_images, train_labels, train_img_names, train_class_name)
            self.valid = Dataset(validation_images, validation_labels, validation_img_names, validation_class_name)
            data_sets = (self.train, self.valid)
        return data_sets

    def load_train(self, train_path, image_size, classes=None, num_objects=None):
        '''
        Description: Reads files from different class folders, resizes them, and saves the pixels and labels
        Input: train_path <string> the path to overacrching folder containing all image files,
               classes <list of strings> contains the names of the different image categories,
               image_size <tuple of ints> is the desired width and height of the input image,
        Return: images, labels, img_names, class_name <tuple of lists> ouput pixels and labeling data
        '''

        images = []
        labels = []  # One-hot encoding array
        img_names = []  # img file base path, the png file
        class_name = []  # Classes english name

        # Check which object folders are actually in the train_path
        if not classes:
            classes = []
            for cls in os.listdir(train_path):
                classes.append(cls)

        for field in classes:
            # Determine number of objects that were captured per object class
            if not num_objects:
                num_objects = 0
                for _ in os.listdir(train_path + '/' + field):
                    num_objects += 1
            print("num objects", num_objects)
            index = classes.index(field)
            
            print('Now going to read {} files (Index: {})'.format(field, index))
            for i in range(1, num_objects + 1):
                img = field + '_' + str(i) + '/images/'
                path = os.path.join(train_path, field, img)
                print("path", path)
                files = glob.glob(path + '*')

                for img in files:
                    image = cv2.imread(img)
                    image = cv2.resize(image, (image_size[0], image_size[1]), 0, 0, cv2.INTER_LINEAR)
                    image = image.astype(np.float32)
                    image = np.multiply(image, 1.0 / 255.0)
                    images.append(image)
                    label = np.zeros(len(classes))
                    label[index] = 1.0
                    labels.append(label)
                    flbase = os.path.basename(img)  # Name of image
                    img_names.append(flbase)
                    class_name.append(field)
        images = np.array(images)
        labels = np.array(labels)
        img_names = np.array(img_names)
        class_name = np.array(class_name)
        if len(class_name) and len(labels):
            print("class_name: {} labels: {}".format(class_name[0], labels[0]))

        return images, labels, img_names, class_name


class Dataset():
    '''
    Description: Contains a numpy matrix of images, this class defines functions that
                    access the dataset.
    '''

    def __init__(self, images, labels, img_names, class_name):
        self.num_examples = images.shape[0]
        self.images = images
        self.labels = labels
        self.img_names = img_names
        self.class_name = class_name
        self.epochs_complete = 0
        self.index_in_epoch = 0
        print("labels: ", labels[1], "END")
        print("img names: ", img_names[1], "END")
        print("class names: ", class_name[1], "END")

    def next_batch(self, batch_size):
        '''
        Description: This function iterates through the images.
        Input: batch_size <int> the number of images in a batch
        Return: The next batch_size of examples from the dataset.
        '''
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            self.epochs_complete += 1
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.images[start:end], self.labels[start:end]
