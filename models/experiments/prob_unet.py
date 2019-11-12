import torch
import torch.nn as nn
from models.probabilistic_unet import ProbabilisticUnet

experiment_name = 'ProbabilisticUnet'
log_dir_name = 'lidc'

# number of filter for the latent levels, they will be applied in the order as loaded into the list
filter_channels = [32, 64, 128, 192]
latent_levels = len(filter_channels) - 1
nclasses = 1
no_convs_fcomb = 4
beta = 10.0 # for loss function
#
use_reversible = False

# use 1 for grayscale, 3 for RGB images
input_channels = 1

epochs_to_train = 5

# model
model = ProbabilisticUnet
