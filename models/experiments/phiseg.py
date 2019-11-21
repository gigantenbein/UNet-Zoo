import torch
import torch.nn as nn
from models.phiseg import PHISeg

experiment_name = 'PHISeg'
log_dir_name = 'lidc'

# number of filter for the latent levels, they will be applied in the order as loaded into the list
filter_channels = [32, 64, 128, 192]
latent_levels = len(filter_channels) - 1

n_classes = 2

no_convs_fcomb = 4 # not used
beta = 10.0 # not used
#
use_reversible = False
exponential_weighting = True

# use 1 for grayscale, 3 for RGB images
input_channels = 1
epochs_to_train = 20
batch_size = [12, 1, 1]

validation_samples = 16

logging_frequency = 100
validation_frequency = 1000

# model
model = PHISeg
