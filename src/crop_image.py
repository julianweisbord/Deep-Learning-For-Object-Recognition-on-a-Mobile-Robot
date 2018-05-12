
'''
Created on March 15th, 2018
author: Michael Rodriguez
sources: https://www.blog.pythonlibrary.org/2017/10/03/how-to-crop-a-photo-with-python/
description: Crops images of a given object class to feed into neural network.
'''
# !/usr/bin/env python
from os import listdir
from os.path import splitext
from PIL import Image
import os
import sys
import argparse

# Definitions and Constants
MIN_NUM_ARGS = 112


def crop(image_path, coords, saved_location):
    """
    crop - Image cropping function
    This function creates an image object to manipulate and
    crop to the desired dimensions. It then saves it in the
    path that it was given.
    # Arguments
        image_path: The path to the image to edit
        coords: A tuple of x/y coordinates (x1, y1, x2, y2)
        saved_location: Path to save the cropped image
    # Return
        Outputs cropped images to given saved image path.
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)


def main():
    '''
    main - Image cropping module
    This function crops images for the number image classes of a given object.
    The function takes an entire image class folder and loops through all
    instances within it cropping each one and saving it in a new directory.
    # Arguments
        None
    # Returns
        Output: None
    '''

    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Image Cropping')

    parser.add_argument('-c', '--class',
                        dest='class_name', action='store',
                        help='Object class name: mug, stapler, book, etc...')

    parser.add_argument('-p', '--path',
                        dest='class_path', action='store',
                        help='Path to data_captured folder containing image classes')

    parser.add_argument('-d', '--dims',
                        dest='image_dims', action='store',
                        help='Desired dimensions to crop image to. Ex. 200 gives you 200x200')

    args = parser.parse_args()

    if (len(sys.argv) < MIN_NUM_ARGS):
        print("Error, invalid number of args, type -h for help")
        exit()

    # Initialize variables
    object_class = args.class_name

    # Directory of where the folder of images is located
    home_dir = args.class_path

    # Full path to image class folder
    image_class_path = home_dir + "data_captured/" + object_class
    image_crop_path = home_dir + "data_captured/cropped/" + object_class

    # This is the desired image dimensions (ex:220x220)
    desired_dims = 200  # Default Value for dimensions if not provided

    if(args.image_dims):
        desired_dims = int(args.image_dims)

    # Get the total number of instances within the image class
    num_folders = len(os.walk(image_class_path).next()[1])

    for instance_num in range(num_folders):

        # Name of the object class to crop
        object_instance = str(instance_num+1)

        text_file_path = image_class_path + "/" + object_class + "_" + object_instance + "/metadata/"
        image_file_path = image_class_path + "/" + object_class + "_" + object_instance + "/images/"
        save_image_path = image_crop_path + "/" + object_class + "_" + object_instance + "/images/"
        index = 0

        # Path dir creation
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)

        # grab all files in directory
        files = [f for f in listdir(text_file_path) if f.endswith(".txt")]
        for file in files:

            path_to_images = image_class_path + "/" + object_class + "_" + object_instance + "/images/"
            # Get the total number of image files within the image class instance
            num_files = len(os.walk(path_to_images).next()[2])

            # Break if we've reached the max number of images in directory
            if index == num_files:
                break

            file_base = splitext(file)[0]
            image_file_name = file_base + ".png"
            print "Object Number: ", object_instance, " Index: ", index  # Printing index of current image
            lines = [line.rstrip('\n') for line in open(text_file_path + file)]

            image_file = image_file_path + image_file_name

            try:
                image_size = lines[1].split()
                height = int(image_size[0])
                width = int(image_size[1])
            except IndexError:
                image_size = 'null'
                index += 1
                continue

            image_filename = save_image_path + file_base + ".png"
            x1 = ((width/2) - (desired_dims/2))
            y1 = ((height/2) - (desired_dims/2))  # x1 & y1 give bottom left corner of pic

            x2 = ((width/2) + (desired_dims/2))
            y2 = ((height/2) + (desired_dims/2))  # x2 & y2 give top right corner of pic

            # Call crop function on the image
            crop(image_file, (x1, y1, x2, y2), image_filename)

            index += 1


if __name__ == "__main__":
    main()
