# Parameter settings
# Forked from github.com/tobybreckon/python-bow-hog-object-detection

import cv2 as cv

# Path to dataset
DATA_TRAINING_PATH_NEG = '.\\dataset\\Train\\WithoutMask'
DATA_TRAINING_PATH_POS = '.\\dataset\\Train\\WithMask'

# Size of the sliding window patch/image patch to be used for classification
DATA_WINDOW_SIZE = [64, 128]

# The maximum left/right, up/down offset to use when generating samples for training
# that are centred around the centre of the image
DATA_WINDOW_OFFSET_FOR_TRAINING_SAMPLES = 3

# Number of sample patches to extract from each negative training example
DATA_TRAINING_SAMPLE_COUNT_NEG = 10

# Number of sample patches to extract from each positive training example
DATA_TRAINING_SAMPLE_COUNT_POS = 5

# Class names
DATA_CLASS_NAMES = {
    'without_mask': 0,
    'with_mask': 1
}

# Settings for BoVW (Bag of Visual Words) approach
BOVW_SVM_PATH = 'svm_bovw.xml'
BOVW_DICT_PATH = 'bovw_dictionary.npy'

BOVW_DICTIONARY_SIZE = 1000
BOVW_SVM_KERNEL = cv.ml.SVM_RBF

# Iteration settings. Generraly, more iteration, more performance
BOVW_SVM_MAX_TRAINING_ITERATIONS = 1000
BOVW_CLUSTERING_ITERATIONS = 100
BOVW_FIXED_FEATURE_PER_IMAGE_TO_USE = 500

# Specify the feature detection
BOVW_USE_ORB_ALWAYS = False

try:
    if BOVW_USE_ORB_ALWAYS:
        print('Forced used of ORB features, not SIFT')
        raise Exception('force use of ORB')
    
    # SIFT is non-free
    # DETECTOR = cv.SIFT_create(nfeatures=BOVW_FIXED_FEATURE_PER_IMAGE_TO_USE)

    # Option to use SURF -- also non-free
    DETECTOR = cv.SURF_create(nfeatures=BOVW_FIXED_FEATURE_PER_IMAGE_TO_USE)

    # SIFT/SURF feature descriptors are floating point -- use KD_TREE approach
    _algorithm = 0      # FLANN_INDEX_KDTREE
    _index_params = dict(algorithm=_algorithm, trees=5)
    _search_params = dict(checks=50)

except:
    # ORB is SIFT/SURF alternative
    DETECTOR = cv.ORB_create(nfeatures=BOVW_FIXED_FEATURE_PER_IMAGE_TO_USE)

    # ORB feature descriptors are integers -- use HASHING approach
    _algorithm = 6      # FLANN_INDEX_LSH
    _index_params = dict(algorithm=_algorithm,
                         table_number=6,
                         key_size=12,
                         multi_probe_level=1)
    _search_params = dict(checks=50)

    if(not(BOVW_USE_ORB_ALWAYS)):
        print('Falling back to using features: ', DETECTOR.__class__())
        BOVW_USE_ORB_ALWAYS = True

print('For BoVW, use feature: ', DETECTOR.__class__())

# Pick matcher based on choice
MATCHER = cv.FlannBasedMatcher(_index_params, _search_params)