import torch
import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse
# from sklearn.metrics import f1_score, classification_report, confusion_matrix
# from medpy.metric import dc, assd, hd

import utils

# if not sys_config.running_on_gpu_host:
#     import matplotlib.pyplot as plt

from train_model import load_data_into_loader

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}


def main(model_path, exp_config, sys_config, do_plots=False):

    n_samples = 50
    model_selection = 'best_ged'

    # Get Data
    net = exp_config.model(input_channels=exp_config.input_channels,
                           num_classes=1,
                           num_filters=exp_config.filter_channels,
                           latent_dim=2,
                           no_convs_fcomb=4,
                           beta=10.0,
                           reversible=exp_config.use_reversible
                           )

    model_name = exp_config.experiment_name + '.pth'
    save_model_path = os.path.join(sys_config.project_root, 'models', model_name)

    net.load_state_dict(torch.load(save_model_path))
    net.eval()

    _, data = load_data_into_loader(sys_config)

    ged_list = []
    ncc_list = []

    for ii, (patch, mask, _) in enumerate(data):

        if ii % 10 == 0:
            logging.info("Progress: %d" % ii)

        net.forward(patch, mask, training=False)
        sample = net.sample(testing=True)
        ground_truth_label = mask

        ged = utils.generalised_energy_distance(sample, ground_truth_label, 1, label_range=range(1,1))
        ged_list.append(ged)

        #ncc = utils.variance_ncc_dist(sample, ground_truth_label)
        #ncc_list.append(ncc)



    ged_arr = np.asarray(ged_list)
    ncc_arr = np.asarray(ncc_list)

    logging.info('-- GED: --')
    logging.info(np.mean(ged_arr))
    logging.info(np.std(ged_arr))

    logging.info('-- NCC: --')
    logging.info(np.mean(ncc_arr))
    logging.info(np.std(ncc_arr))

    np.savez(os.path.join(model_path, 'ged%s_%s.npz' % (str(n_samples), model_selection)), ged_arr)
    np.savez(os.path.join(model_path, 'ncc%s_%s.npz' % (str(n_samples), model_selection)), ncc_arr)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a network on the test dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    parser.add_argument("LOCAL", type=str, help="Is this script run on the local machine or the BIWI cluster?")
    args = parser.parse_args()

    if args.LOCAL == 'local':
        print('Running with local configuration')
        import config.local_config as sys_config
    else:
        import config.system as sys_config

    base_path = sys_config.project_root

    model_path = args.EXP_PATH
    config_file = args.EXP_PATH
    #config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    main(model_path, exp_config=exp_config, sys_config=sys_config, do_plots=False)