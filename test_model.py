import torch
from torchvision.utils import save_image
import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse
# from sklearn.metrics import f1_score, classification_report, confusion_matrix
from medpy.metric import dc, assd, hd

import utils

# if not sys_config.running_on_gpu_host:
#     import matplotlib.pyplot as plt

from train_model import load_data_into_loader

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}


def test_quantitative(model_path, exp_config, sys_config, do_plots=False):

    n_samples = 50
    model_selection = 'best_ged'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get Data
    net = exp_config.model(input_channels=exp_config.input_channels,
                           num_classes=2,
                           num_filters=exp_config.filter_channels,
                           reversible=exp_config.use_reversible
                           )

    net.to(device)

    model_name = exp_config.experiment_name + '.pth'
    save_model_path = os.path.join(sys_config.project_root, 'models', model_name)

    map = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.load_state_dict(torch.load('/Users/marcgantenbein/PycharmProjects/UNet-Zoo/models/Unet.pth', map_location=map))
    net.eval()

    _, data = load_data_into_loader(sys_config)

    ged_list = []
    dice_list = []
    ncc_list = []

    ged = 0
    with torch.no_grad():
        for ii, (patch, mask, _, masks) in enumerate(data):
            print('Step: {}'.format(ii))
            patch.to(device)
            mask.to(device)
            if ii % 10 == 0:
                logging.info("Progress: %d" % ii)
                print("Progress: {} GED: {}".format(ii, ged))

            net.forward(patch, mask=mask, training=False)
            sample = net.sample(testing=True)
            ground_truth_labels = masks.view(-1,1,128,128)

            ged = utils.generalised_energy_distance(sample, ground_truth_labels, 4, label_range=range(1, 5))
            print(ged)
            ged_list.append(ged)

            dice_list.append(dc(sample.view(-1, 128, 128).detach().numpy(), mask.view(-1, 128, 128).detach().numpy()))

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

   # np.savez(os.path.join(model_path, 'ged%s_%s.npz' % (str(n_samples), model_selection)), ged_arr)
   # np.savez(os.path.join(model_path, 'ncc%s_%s.npz' % (str(n_samples), model_selection)), ncc_arr)

def test_segmentation(exp_config, sys_config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get Data
    net = exp_config.model(input_channels=exp_config.input_channels,
                           num_classes=2,
                           num_filters=exp_config.filter_channels,
                           latent_dim=exp_config.latent_levels,
                           no_convs_fcomb=4,
                           beta=10.0,
                           reversible=exp_config.use_reversible
                           )

    net.to(device)

    model_name = exp_config.experiment_name + '.pth'
    save_model_path = os.path.join(sys_config.project_root, 'models', model_name)

    map = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.load_state_dict(torch.load(save_model_path, map_location=map))
    net.eval()

    _, data = load_data_into_loader(sys_config)

    with torch.no_grad():
        for ii, (patch, mask, _, masks) in enumerate(data):

            if ii % 10 == 0:
                logging.info("Progress: %d" % ii)
                print("Progress: {}".format(ii))

            net.forward(patch, mask, training=False)
            sample = torch.nn.functional.softmax(net.sample(testing=True))

            n = min(patch.size(0), 8)
            comparison = torch.cat([patch[:n],
                                     masks.view(-1, 1, 128, 128),
                                     sample[0][1].view(-1,1,128,128)[:n]])
            #comparison = sample.view(-1, 1, 128, 128)
            save_image(comparison.cpu(),
                       'segmentation/' + exp_config.experiment_name + '/comp_' + str(ii) + '.png', nrow=n)


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

    utils.makefolder(os.path.join(sys_config.project_root, 'segmentation/', exp_config.experiment_name))

    #test_quantitative(model_path, exp_config, sys_config)
    test_segmentation(exp_config, sys_config)
    #main(model_path, exp_config=exp_config, sys_config=sys_config, do_plots=False)