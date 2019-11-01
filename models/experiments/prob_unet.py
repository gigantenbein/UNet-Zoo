import torch
import torch.nn as nn

experiment_name = 'ProbabilisticUnet'

# number of filter for the latent levels, they will be applied in the order as loaded into the list
filter_channels = [32, 64, 128, 192]
latent_levels = len(filter_channels) - 1

#
use_reversible = False

# use 1 for grayscale, 3 for RGB images
input_channels = 1

epochs_to_train = 10

