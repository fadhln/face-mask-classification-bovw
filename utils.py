# Utility functions for BoVW algorithms
# Forked from github.com/tobybreckon/python-bow-hog-object-detection

import os
import numpy as np
import cv2 as cv
from numpy.core.shape_base import stack
import params
import math
import random

# Global flags to facilitate output of additional info per stage/function
show_additional_process_information = False
show_images_as_they_are_loaded = False
show_images_as_they_are_sampled = False

# Timing information -- for training
# Helper function for timing code execution
def get_elapsed_time(start):
    return(cv.getTickCount() - start) / cv.getTickFrequency()

def format_time(time):
    time_str = ''
    if time < 60.0:
        time_str = "{}s".format(round(time, 1))
    elif time > 60.0:
        minutes = time / 60.0
        time_str = '{}m:{}s'.format(int(minutes), round(time % 60, 2))
    return time_str

def print_duration(start):
    time = get_elapsed_time(start)
    print(('Took {}'.format(format_time(time))))

# Reads all the images in a given folder path and returns the results
# This will become memory intensive
def read_all_images(path):
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    for image_path in images_path:
        # Add-in a check to skip non-jpg or non-png named files
        if (('.png' in image_path) or ('.jpg' in image_path)):
            img = cv.imread(image_path)
            images.append(img)
            if show_additional_process_information:
                print('loading file - ', image_path)
        else:
            if show_additional_process_information:
                print('skipping non-png/non-jpg file - ', image_path)
    return images

# Stack array (np.vstack()) of items as basic Python data manipulation
def stack_array(arr):
    stacked_arr = np.array([])
    for item in arr:
        # Only stack if it is not empty
        if len(item) > 0:
            if len(stacked_arr) == 0:
                stacked_arr = np.array(item)
            else:
                stacked_arr = np.vstack((stacked_arr, item))
    return stacked_arr

# Transform between class numbers (i.e. codes) - {0,1,2,3,...,N} to
# classnames {'dog', 'cat', 'cow', ...}
def get_class_number(class_name):
    return params.DATA_CLASS_NAMES.get(class_name, 0)

def get_class_name(class_code):
    for name, code in params.DATA_CLASS_NAMES.items():
        if code == class_code:
            return name

# Image data class object that contain the images, descriptors, and BoVW histogram
class ImageData(object):
    def __init__(self, img):
        self.img = img
        self.class_name = ''
        self.class_number = None
        self.bovw_descriptors = np.array([])
    
    def set_class(self, class_name):
        self.class_name = class_name
        self.class_number = get_class_number(self.class_name)
        if show_additional_process_information:
            print('class name : ', self.class_name, ' - ', self.class_number)
    
    def compute_bovw_descriptor(self):
        # Generate the feature descriptors for a given image
        self.bovw_descriptors = params.DETECTOR.detectAndCompute(self.img, None)[1]

        if self.bovw_descriptors is None:
            self.bovw_descriptors = np.array([])
        
        if show_additional_process_information:
            print('# feature descriptor computed - ', len(self.bovw_descriptors))
    
    def generate_bovw_hist(self, dictionary):
        self.bovw_histogram = np.zeros((len(dictionary), 1))

        # Generate the BoVW histogram of feature occurance from descriptors
        if(params.BOVW_USE_ORB_ALWAYS):
            # FLANN matcher with ORB needs dictionary to be uint8
            matches = params.MATCHER.match(self.bovw_descriptors, np.uint8(dictionary))
        
        else:
            # FLANN matcher with SIFT/SURF needs descriptor to be type32
            matches = params.MATCHER.match(np.float32(self.bovw_descriptors), dictionary)
        
        for match in matches:
            # Get which visual word this descriptor matches in the dictionary
            # match.trainIdx is the visual_word
            # Hard assignment -- increase count for visual words histogram
            self.bovw_histogram[match.trainIdx] += 1
        
        # [IMPORTANT] - normalize the histogram to L1 to remove bias
        self.bovw_histogram = cv.normalize(self.bovw_histogram, None, alpha=1, beta=0, norm_type=cv.NORM_L1)

# Generates a set of random sample patches from a given image of a specified size
def generate_patches(img, 
                     sample_patches_to_generate=0, 
                     center_weighted=False, 
                     center_sampling_offset=10,
                     patch_size=(64,128)):
    patches = []

    # If no patches specified just return original image
    if(sample_patches_to_generate == 0):
        return [img]
    
    else:
        img_height, img_width, _ = img.shape
        patch_height = patch_size[0]
        patch_width = patch_size[1]
        
        for patch_count in range(sample_patches_to_generate):
            # If we are using center-weighted patches, first grab the center
            # image as the first sample then take the rest around
            if (center_weighted):
                # Compute patch location in the center of the image
                patch_start_h = math.floor(img_height / 2) - math.floor(patch_height / 2)
                patch_start_w = math.floor(img_width / 2) - math.floor(patch_width / 2)

                if (patch_count > 0):
                    patch_start_h = random.randint(patch_start_h - center_sampling_offset, patch_start_h + center_sampling_offset)
                    patch_start_w = random.randint(patch_start_w - center_sampling_offset, patch_start_w + center_sampling_offset)
                
                # Else, get patches randomly from anywhere in the image
            else:
                patch_start_h = random.randint(0, (img_height - patch_height))
                patch_start_w = random.randint(0, (img_width - patch_width))
                
                # Add the patch to the list of patches
            patch = img[patch_start_h:(patch_start_h+patch_height), patch_start_w:(patch_start_w+patch_width)]

            if (show_images_as_they_are_sampled):
                cv.imshow('patch', patch)
                cv.waitkey(5);
                
            patches.insert(patch_count, patch)

        return patches

def generate_dictionary(imgs_data, dictionary_size):
    # Extracting descriptors
    desc = stack_array([img_data.bovw_descriptors for img_data in imgs_data])

    # [IMPORTANT] cv.kmeans() clustering only accept type32 descriptors
    desc = np.float32(desc)
    
    # Perform clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, params.BOVW_CLUSTERING_ITERATIONS, 0.01)
    flags = cv.KMEANS_PP_CENTERS

    # Desc is a type32 numpy array of vstacked descriptors
    compactness, labels, dictionary = cv.kmeans(desc, dictionary_size, None, criteria, 1, flags)
    np.save(params.BOVW_DICT_PATH, dictionary)

    return dictionary

# Add images from a specified path to the dataset, add the corresponding class_name
def load_image_path(path,
                    class_name,
                    imgs_data,
                    samples=0,
                    center_weighting=False,
                    center_sampling_offset=10,
                    patch_size=(64,120)):
    # Read all images at a location
    imgs = read_all_images(path)
    img_count = len(imgs_data)
    for img in imgs:
        if (show_images_as_they_are_loaded):
            cv.imshow('example', img)
            cv.waitKey(5)
        
        # Generate N sample patches for each sample image
        # if zero sample is provided, return the original image

        for img_patch in generate_patches(img, samples, center_weighting, center_sampling_offset, patch_size):
            if show_additional_process_information:
                print('path: ', path, 'class_name: ', class_name, 'patch #: ', img_count)
                print('patch: ', patch_size, 'from center: ', center_weighting, 'with offset: ', center_sampling_offset)
            
            # Add each image patch to the dataset
            img_data = ImageData(img_patch)
            img_data.set_class(class_name)
            imgs_data.insert(img_count, img_data)
            img_count += 1

    return imgs_data

# Load image data from a specified path
def load_images(paths, class_names, sample_set_sizes, use_center_weighting_flags, center_sampling_offset=10, patch_size=(64,120)):
    imgs_data = []

    for path, class_name, sample_count, center_weighting in zip(paths, class_names, sample_set_sizes, use_center_weighting_flags):
        load_image_path(path, class_name, imgs_data, sample_count, center_weighting, center_sampling_offset, patch_size)
    
    return imgs_data

# Return the global set of BoVW histogram for the dataset of images
def get_bovw_histograms(imgs_data):
    samples = stack_array([[img_data.bovw_histogram] for img_data in imgs_data])
    return np.float32(samples)

# Return the global set of numerical class labels for the dataset of images
def get_class_labels(imgs_data):
    class_labels = [img_data.class_number for img_data in imgs_data]
    return np.int32(class_labels)