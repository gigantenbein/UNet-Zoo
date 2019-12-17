import torch
import torch.nn as nn
from models.phiseg3D import PHISeg3D
from data.lidc_data import lidc_data
from utils import normalise_image

experiment_name = 'PHISeg_brats'
log_dir_name = 'lidc'

data_loader=lidc_data

# number of filter for the latent levels, they will be applied in the order as loaded into the list
filter_channels = [32, 64, 128, 192, 192, 192, 192]
latent_levels = 5

iterations = 5000000

n_classes = 2
num_labels_per_subject = 4

no_convs_fcomb = 4 # not used
beta = 10.0 # not used
#
use_reversible = False
exponential_weighting = True

# use 1 for grayscale, 3 for RGB images
input_channels = 1
epochs_to_train = 20
batch_size = 12
image_size = 0

augmentation_options = {'do_flip_lr': True,
                        'do_flip_ud': True,
                        'do_rotations': True,
                        'do_scaleaug': True,
                        'nlabels': n_classes}

input_normalisation = normalise_image

validation_samples = 16
num_validation_images = 100

logging_frequency = 1000
validation_frequency = 1000

weight_decay = 10e-5

pretrained_model = None #'PHISeg_best_ged.pth'
# model
model = PHISeg3D

CHANNELS = [60, 120, 240, 360, 480]
INITIAL_LR = 1e-4
L2_REGULARIZER = 1e-5

#logging settings
LOG_EVERY_K_ITERATIONS = 5 #0 to disable logging
LOG_MEMORY_EVERY_K_ITERATIONS = False
LOG_MEMORY_EVERY_EPOCH = True
LOG_EPOCH_TIME = True
LOG_VALIDATION_TIME = True
LOG_HAUSDORFF_EVERY_K_EPOCHS = 0 #must be a multiple of VALIDATE_EVERY_K_EPOCHS
LOG_COMETML = False
LOG_PARAMCOUNT = True
LOG_LR_EVERY_EPOCH = True

#data and augmentation
TRAIN_ORIGINAL_CLASSES = False #train on original 5 classes
DATASET_WORKERS = 1
SOFT_AUGMENTATION = False #Soft augmetation directly works on the 3 classes. Hard augmentation augments on the 5 orignal labels, then takes the argmax
NN_AUGMENTATION = True #Has priority over soft/hard augmentation. Uses nearest-neighbor interpolation
DO_ROTATE = True
DO_SCALE = True
DO_FLIP = True
DO_ELASTIC_AUG = True
DO_INTENSITY_SHIFT = True
#RANDOM_CROP = [128, 128, 128]

ROT_DEGREES = 20
SCALE_FACTOR = 1.1
SIGMA = 10
MAX_INTENSITY_SHIFT = 0.1
