'''
Created on February 1st, 2018
author: Julian Weisbord
sources: https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier
description: Module to prepare image data for tensorflow model.
'''

import os
import glob
import numpy as np
from sklearn.utils import shuffle
import cv2

# Plan:
# if data is already in a file (first line of label_file is [last_data_change_time, last time file was written to])
    # if last data change made > last time label_file was written to
    #     go through all of the image data and write to label_file with new data changes
    # else
    #     use label_file for training
# else
    # train network by parsing images in folders
    # go through all of the image data and write to label_file with new data changes

class PrepareData():
    def __init__(self):
        self.train = None
        self.valid = None


    def read_train_sets(self, train_path, classes, validation_size):

        images, labels, img_names, cls = self.load_train(train_path, classes)
        images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  # Randomize arrays

        if isinstance(validation_size, float):
            validation_size = int(validation_size * images.shape[0])

        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]
        validation_img_names = img_names[:validation_size]
        validation_cls = cls[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]
        train_img_names = img_names[validation_size:]
        train_cls = cls[validation_size:]

        self.train = Dataset(train_images, train_labels, train_img_names, train_cls)
        self.valid = Dataset(validation_images, validation_labels, validation_img_names, validation_cls)
        data_sets = (self.train, self.valid)
        return data_sets

    def load_train(self, train_path, classes):
        images = []
        labels = []
        img_names = []
        cls = []

        print('Going to read training images')
        for fields in classes:
            index = classes.index(fields)
            print('Now going to read {} files (Index: {})'.format(fields, index))
            img1 = fields + '_1'  # Just do first images for now
            path = os.path.join(train_path, fields, img1, "images", fields)
            files = glob.glob(path + '*')
            # print('files:\n', files)
            for fl in files:
                image = cv2.imread(fl)
                # cv2.imshow(fl,image)  # Test
                # k = cv2.waitKey(0)  # Test
                # image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                image = image.astype(np.float32)
                image = np.multiply(image, 1.0 / 255.0)
                images.append(image)
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                flbase = os.path.basename(fl)  # Name of image
                img_names.append(flbase)
                cls.append(fields)
        images = np.array(images)
        labels = np.array(labels)
        img_names = np.array(img_names)
        cls = np.array(cls)

        return images, labels, img_names, cls

class Dataset():
    def __init__(self, images, labels, img_names, cls):
        self.num_examples = images.shape[0]
        self.images = images
        self.labels = labels
        self.img_names = img_names
        self.cls = cls
        self.epochs_complete = 0
        self.index_in_epoch = 0

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.num_examples:
            self.epochs_complete += 1
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch

        # return self.images[start:end], self.labels[start:end], self.img_names[start:end], self.cls[start:end]
        return self.images[start:end], self.labels[start:end]
