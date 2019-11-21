import torch
import numpy as np

import torchvision
from torchvision.utils import save_image
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Python bundle packages
import os
import logging
import shutil
from importlib.machinery import SourceFileLoader
import argparse
import time
from medpy.metric import dc
import math

# own files
from load_LIDC_data import load_data_into_loader, create_pickle_data_with_n_samples
import utils
from test_model import test_segmentation

# catch all the warnings with the debugger
# import warnings
# warnings.filterwarnings('error')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# TODO: check phiseg loss function, i.e. torch.mean for the reconstruction loss
# TODO: run PHiSeg without BN and different weigths for BN
# TODO: create returning of a sample arrangement and adapt the GED and NCC functions

class UNetModel:
    '''Wrapper class for different Unet models to facilitate training, validation, logging etc.
        Args:
            exp_config: Experiment configuration file
    '''
    def __init__(self, exp_config):

        self.net = exp_config.model(input_channels=exp_config.input_channels,
                                    num_classes=exp_config.n_classes,
                                    num_filters=exp_config.filter_channels,
                                    latent_dim=exp_config.latent_levels,
                                    no_convs_fcomb=exp_config.no_convs_fcomb,
                                    beta=exp_config.beta,
                                    reversible=exp_config.use_reversible
                                    )
        self.exp_config = exp_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', min_lr=1e-6, verbose=True, patience=100)

        self.writer = SummaryWriter()
        self.epochs = exp_config.epochs_to_train

        self.dice_mean = 0
        self.val_loss = 0

    def train(self, train_loader, validation_loader):
        self.net.train()
        logging.info('Starting training.')

        for self.epoch in range(self.epochs):
            for self.step, (patch, mask, _, masks) in enumerate(train_loader):
                patch = patch.to(self.device)
                mask = mask.to(self.device)  # N,H,W
                mask = torch.unsqueeze(mask, 1)  # N,1,H,W
                masks = masks.to(self.device)

                self.mask = mask
                self.patch = patch
                self.masks = masks

                self.net.forward(patch, mask, training=True)
                self.loss = self.net.loss(mask)
                assert math.isnan(self.loss) == False

                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                if self.step % exp_config.logging_frequency == 0:
                    logging.info('Epoch {} Step {} Loss {}'.format(self.epoch, self.step, self.loss))
                    logging.info('Epoch: {} Number of processed patches: {}'.format(self.epoch, self.step))
                    print('Epoch {} Step {} Loss {}'.format(self.epoch, self.step, self.loss))
                    print('Epoch: {} Number of processed patches: {}'.format(self.epoch, self.step))
                    self._create_tensorboard_summary()
                if self.step % exp_config.validation_frequency == 0:
                    self.validate(validation_loader)
                    pass

                self.scheduler.step(self.loss)

            logging.info('Finished epoch {}'.format(self.epoch))
            print('Finished epoch {}'.format(self.epoch))
        logging.info('Finished training.')
        logging.info('Starting validation')
        #self.validate()

    def save_model(self):
        model_name = self.exp_config.experiment_name + '.pth'
        save_model_path = os.path.join(sys_config.project_root, 'models', model_name)
        torch.save(self.net.state_dict(), save_model_path)
        logging.info('saved model to .pth file in {}'.format(save_model_path))

    def test(self):
        # test_quantitative(model_path, exp_config, sys_config)
        test_segmentation(exp_config, sys_config, 10)

    def validate(self, validation_loader):
        with torch.no_grad():
            self.net.eval()
            logging.info('Validation for step {}'.format(self.step))

            ged_list = []
            dice_list = []
            ncc_list = []

            time_ = time.time()

            for val_step, (val_patch, val_mask, _, val_masks) in enumerate(validation_loader):
                val_patch = val_patch.to(self.device)
                val_mask = val_mask.to(self.device)  # N,H,W
                val_mask = torch.unsqueeze(val_mask, 1)  # N,1,H,W

                self.net.forward(val_patch, val_mask, training=True)
                self.val_loss = self.net.loss(val_mask)

                sample = torch.sigmoid(self.net.sample(testing=True))
                sample = torch.chunk(sample, 2, dim=1)[0]

                sample_arrangement = [torch.sigmoid(self.net.sample(testing=True)) for i in range(4)]

                # Generalized energy distance
                #ged = utils.generalised_energy_distance(sample_arrangement, self.masks, 4, label_range=range(1, 5))
                #ged_list.append(ged)

                # Dice coefficient
                dice = dc(sample.detach().cpu().numpy(), val_mask.detach().cpu().numpy())
                dice_list.append(dice)

                # Normalised Cross correlation
                #ncc = utils.variance_ncc_dist(sample.numpy(), val_masks.numpy())
                #ncc_list.append(ncc)

            #self.validation_dice_mean = dice_list.mean()
            #self.validation_ged_mean = ged_list.mean()
            #self.validation_ncc_mean = ncc_list.mean()

            logging.info('Validation took {} seconds'.format(time.time()-time_))
            self.dice_mean = np.asarray(dice_list).mean()
            logging.info(self.dice_mean)

            self.net.train()

    def _create_tensorboard_summary(self):
        with torch.no_grad():

            # plot images of current patch for summary
            sample = torch.sigmoid(self.net.sample())
            sample = torch.chunk(sample, 2, dim=1)[0]

            #sample = torch.round(sample)

            self.writer.add_image('Patch/GT/Sample_from_Epoch_{}'.format(self.epoch),
                                  torch.cat([self.patch, self.mask.view(-1, 1, 128, 128),
                                             sample], dim=2), global_step=self.step, dataformats='NCHW')
            # self.writer.add_image('Deep_Supervision_from_Epoch_{}'.format(self.epoch),
            #                       forward_pass, global_step=self.step, dataformats='NCHW')
            # add current loss
            self.writer.add_scalar('Loss_of_current_batch_from_Epoch_{}'.format(self.epoch),
                                   self.loss, global_step=self.step)
            self.writer.add_scalar('Dice_score_of_last_validation_from_Epoch_{}'.format(self.epoch),
                                   self.dice_mean, global_step=self.step)
            self.writer.add_scalar('Validation_loss_of_last_validation_from_Epoch_{}'.format(self.epoch),
                                   self.val_loss, global_step=self.step)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment config file")
    parser.add_argument("LOCAL", type=str, help="Is this script run on the local machine or the BIWI cluster?")
    parser.add_argument("dummy", type=str, help="Is the module run with dummy training?")
    args = parser.parse_args()

    config_file = args.EXP_PATH
    config_module = config_file.split('/')[-1].rstrip('.py')

    if args.LOCAL == 'local':
        print('Running with local configuration')
        import config.local_config as sys_config
        import matplotlib.pyplot as plt
    else:
        import config.system as sys_config

    logging.info('Running experiment with script: {}'.format(config_file))

    exp_config = SourceFileLoader(config_module, config_file).load_module()

    log_dir = os.path.join(sys_config.log_root, exp_config.log_dir_name, exp_config.experiment_name)

    utils.makefolder(log_dir)

    shutil.copy(exp_config.__file__, log_dir)
    logging.info('!!!! Copied exp_config file to experiment folder !!!!')

    logging.info('**************************************************************')
    logging.info(' *** Running Experiment: %s', exp_config.experiment_name)
    logging.info('**************************************************************')

    model = UNetModel(exp_config)
    if args.dummy == 'dummy':
        #create_pickle_data_with_n_samples(sys_config,1000)
        train_loader, test_loader, validation_loader = load_data_into_loader(
            sys_config, 'size1000/', batch_size=exp_config.batch_size)
        utils.makefolder(os.path.join(sys_config.project_root, 'segmentation/', exp_config.experiment_name))
        model.train(train_loader, validation_loader)
    else:
        train_loader, test_loader, validation_loader = load_data_into_loader(
            sys_config, '', batch_size=exp_config.batch_size)
        utils.makefolder(os.path.join(sys_config.project_root, 'segmentation/', exp_config.experiment_name))
        model.train(train_loader, validation_loader)
        model.save_model()

    model.save_model()
