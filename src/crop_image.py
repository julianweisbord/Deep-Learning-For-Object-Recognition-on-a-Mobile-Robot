#!/usr/bin/env python

from os import listdir
from os.path import isfile, join, splitext
from PIL import Image
import numpy as np
import cv2
import glob


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)


'''
    Main - Image cropping module
    This function crops images for the 5 image classes of a given object. The
    function takes an entire image class folder and loops through all 5
    instances within it cropping each one and saving it in a new directory.
    # Arguments
        None
    # Returns
        Output: None
'''


def main():

    for instance_num in range(5):

        # Name of the object class to crop
        object_class = "book"
        object_instance = str(instance_num+1)

        # Directory of where the folder of images is located
        home_dir = "/Volumes/SAN_MINI/"

        text_file_path = home_dir + "data_captured/" + object_class + "/" + object_class + "_" + object_instance + "/metadata/"
        image_file_path = home_dir + "data_captured/" + object_class + "/" + object_class + "_" + object_instance + "/images/"
        save_image_path = home_dir + "data_captured/cropped/" + object_class + "/" + object_class + "_" + object_instance + "/images/"
        desired_dims = 220  # This is the desired image dimensions (ex:220x220)
        index = 0

        # grab all files in directory
        files = [f for f in listdir(text_file_path) if f.endswith(".txt")]
        for file in files:

            # Break if we've reached the number of images in directory
            if index == 80:
                break

            file_base = splitext(file)[0]
            image_file_name = file_base + ".png"
            print("Index: ", index)  # Printing index of current current image
            lines = [line.rstrip('\n') for line in open(text_file_path + file)]

            image_file = image_file_path + image_file_name
            image_size = lines[1].split()
            height = int(image_size[0]) 
            width = int(image_size[1])

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
