'''
Created on March 10th, 2018
author: Julian Weisbord
sources: http://www.image-net.org/
        https://github.com/fchollet/deep-learning-models/blob/master/inception_resnet_v2.py
description: Inception ResNet Convolutional Neural Network implementation that takes
                different images and classifies them into 5 different categories.
'''

# External Imports
import sys
import os
import time
import glob
import warnings
import shutil
import numpy as np
from keras import backend as K
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.models import model_from_yaml
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, History
from keras.applications.resnet50 import preprocess_input

import model as mdl

BASE_WEIGHT_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/'
CLASSES = ['book', 'chair', 'mug', 'screwdriver', 'stapler']
DEFAULT_TRAIN_PATH = '../image_data/train'
DEFAULT_VAL_PATH = '../image_data/validation'
N_CLASSES = len(CLASSES)
IMAGE_HEIGHT = 139
IMAGE_WIDTH = 139
BATCH_SIZE = 10
N_EPOCHS = 11
N_SAMPLES = 1757
INCLUDE_TOP = True


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    '''Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    '''
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    '''Adds a Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`

    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch. Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names. The Inception-ResNet blocks
            are repeated many times in this network. We use `block_idx` to identify
            each of the repetitions. For example, the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`, ane the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](keras./activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).

    # Returns
        Output tensor for the block.

    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    '''
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x




def InceptionResNetV2(include_top=True,
                      weights='imagenet',
                      weights_path=None,
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000):
    '''Instantiates the Inception-ResNet v2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that when using TensorFlow, for best performance you should
    set `"image_data_format": "channels_last"` in your Keras config
    at `~/.keras/keras.json`.

    Note that the default input image size for this model is 299x299, instead
    of 224x224 as in the VGG16 and ResNet models. Also, the input preprocessing
    function is different (i.e., do not use `imagenet_utils.preprocess_input()`
    with this model. Use `preprocess_input()` defined in this module instead).

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or `'imagenet'` (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional layer.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.

    # Returns
        A Keras `Model` instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with an unsupported backend.
    '''
    if K.backend() in {'cntk'}:
        raise RuntimeError(K.backend() + ' backend is currently unsupported for this model.')

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid')
    x = conv2d_bn(x, 32, 3, padding='valid')
    x = conv2d_bn(x, 64, 3)
    x = MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, padding='valid')
    x = MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = Model(inputs, x, name='inception_resnet_v2')

    print("Loading Weights")

    # Load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    model.load_weights(weights_path)
    print("Weights loaded successfully")

    model.layers.pop()
    print("Number of layers: ", len(model.layers))
    for layer in model.layers:
        layer.trainable = True
    x = Dense(N_CLASSES, activation='softmax', name='predictions')(model.layers[-1].output)
    x.trainable = True

    model = Model(inputs, x)
    print("Saving Weights: ")
    model.save_weights('saved_weights.h5')

    return model

def setup_image_dir(train_path, val_path):
    '''
    Description: Takes the directory structure for training and validation images used
                    in model.py and converts them into the structure that keras expects
                    at the locations: train_path, val_path.
    Input: train_path <string> path to directory where training data should be stored.
               val_path <string> path to directory where validation data should be stored.
    Return: None
    '''

    files = glob.glob(train_path + '/*')
    # Create file directory structure
    print("len files: ", len(files))
    if not files:
        # Check if there is an image directory that model.py uses
            # then copy those images
        if os.path.exists(mdl.DEFAULT_TRAIN_PATH):
            print("mdl's train path exists")
            for field in CLASSES:
                for i in range(1, N_CLASSES + 1):
                    img = field + '_' + str(i) + '/images/'
                    path = os.path.join(mdl.DEFAULT_TRAIN_PATH, field, img)
                    print("path", path)
                    files = glob.glob(path + '*')
                    train_dest = train_path + "/" + field + "/"
                    val_dest = val_path + "/" + field + "/"
                    print("base dest: ", train_dest)
                    if not os.path.exists(train_dest):
                        os.mkdir(train_dest)
                    if not os.path.exists(val_dest):
                        os.mkdir(val_dest)
                    for fl in files:
                        print("file", fl)
                        shutil.copy(fl, train_dest)
                        shutil.copy(fl, val_dest)

def main():
    val_path = DEFAULT_VAL_PATH
    if len(sys.argv) < 2:
        print("Using default training dataset path")
        train_path = DEFAULT_TRAIN_PATH
    else:
        train_path = sys.argv[1]
        if len(sys.argv) == 3:
            val_path = sys.argv[2]

    setup_image_dir(train_path, val_path)  # Check if the image directory structure exists
    # Locate and load weights:
    n_classes = N_CLASSES
    weights_path = "./saved_weights.h5"
    if not os.path.exists(weights_path):
        weights_path = "./inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5"
        if not os.path.exists("./inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5"):
            if INCLUDE_TOP is True:
                weights_filename = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
                weights_path = get_file(weights_filename,
                                        BASE_WEIGHT_URL + weights_filename,
                                        cache_subdir='models',
                                        md5_hash='e693bd0210a403b3192acc6073ad2e96')
            else:
                weights_filename = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
                weights_path = get_file(weights_filename,
                                        BASE_WEIGHT_URL + weights_filename,
                                        cache_subdir='models',
                                        md5_hash='d19885ff4a710c122648d3b5c3b684e4')
        n_classes = 1000

    model = InceptionResNetV2(include_top=True, weights='imagenet',
                              weights_path=weights_path,
                              classes=n_classes)

    # prepare data augmentation configuration
    print("Passing images through model")
    train_datagen = image.ImageDataGenerator(
        # rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = image.ImageDataGenerator() #rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        DEFAULT_TRAIN_PATH,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        DEFAULT_VAL_PATH,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')


    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  loss_weights=None,
                  sample_weight_mode=None)

    start_time = time.time()
    model.fit_generator(
        train_generator,
        steps_per_epoch=N_SAMPLES // BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=validation_generator,
        validation_steps=N_SAMPLES // BATCH_SIZE)
    end_time = time.time() - start_time
    print("Total time", end_time)
    # Save the model to a yaml file
    print("Saving model to yaml")
    model_yaml = model.to_yaml()
    with open("robot-environment-model/in_resnet.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)


if __name__ == '__main__':
    main()
