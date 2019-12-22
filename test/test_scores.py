"""Testing scoring functions"""

import pytest
import os
from importlib.machinery import SourceFileLoader
import utils
import shutil
import torch
import math
import matplotlib.pyplot as plt
import torchvision



@pytest.fixture
def lidc_data():

    config_file = '/Users/marcgantenbein/PycharmProjects/UNet-Zoo/models/experiments/phiseg_rev_7_5_12.py'
    config_module = config_file.split('/')[-1].rstrip('.py')

    print('Running with local configuration')
    import config.local_config as sys_config
    import matplotlib.pyplot as plt

    exp_config = SourceFileLoader(config_module, config_file).load_module()

    data = exp_config.data_loader(sys_config=sys_config, exp_config=exp_config)
    return data


def test_ncc(lidc_data):
    random_index = 99
    s_gt_arr = lidc_data.test.labels[random_index, ...]

    x_b = lidc_data.test.images[random_index, ...]
    patch = torch.tensor(x_b, dtype=torch.float32).to('cpu')


    assert s_gt_arr.shape == (128, 128, 4)
    val_masks = torch.tensor(s_gt_arr, dtype=torch.float32).to('cpu')  # HWC
    val_masks = val_masks.transpose(0, 2).transpose(1, 2)
    assert val_masks.shape == (4, 128, 128)

    s_gt_arr_r = val_masks.unsqueeze(dim=1)

    ground_truth_arrangement_one_hot = utils.convert_batch_to_onehot(s_gt_arr_r, nlabels=2)

    ncc = utils.variance_ncc_dist(ground_truth_arrangement_one_hot, ground_truth_arrangement_one_hot)

    assert math.isclose(ncc[0], 1.0)


def test_ged(lidc_data):
    pass


def test_dice(lidc_data):
    pass

