import torch
import numpy as np

import torchvision
from torchvision.utils import save_image
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter, FileWriter

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
from data.lidc_data import lidc_data
from load_LIDC_data import load_data_into_loader, create_pickle_data_with_n_samples
import utils
from test_model import test_segmentation
from torchvision.transforms import Normalize

# catch all the warnings with the debugger
# import warnings
# warnings.filterwarnings('error')
from utils import show_tensor
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class UNetModel:
    '''Wrapper class for different Unet models to facilitate training, validation, logging etc.
        Args:
            exp_config: Experiment configuration file as given in the experiment folder
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
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', min_lr=1e-4, verbose=True, patience=5000)

        self.mean_loss_of_epoch = 0
        self.tot_loss = 0
        self.kl_loss = 0
        self.reconstruction_loss = 0
        self.dice_mean = 0
        self.val_loss = 0
        self.foreground_dice = 0

        self.val_recon_loss = 0
        self.val_elbo = 0
        self.val_kl_loss = 0
        self.avg_dice = 0
        self.avg_ged = -1
        self.avg_ncc = -1

        self.best_dice = -1
        self.best_loss = np.inf
        self.best_ged = np.inf
        self.best_ncc = -1

        self.training_writer = SummaryWriter()
        self.validation_writer = SummaryWriter(comment='_validation')
        self.iteration = 0

    def train(self, data):
        self.net.train()
        logging.info('Starting training.')

        for self.iteration in range(1, self.exp_config.iterations):

            x_b, s_b = data.train.next_batch(exp_config.batch_size)

            patch = torch.tensor(x_b, dtype=torch.float32).to(self.device)

            mask = torch.tensor(s_b, dtype=torch.float32).to(self.device)
            mask = torch.unsqueeze(mask, 1)

            self.mask = mask
            self.patch = patch

            self.net.forward(patch, mask, training=True)
            self.loss = self.net.loss(mask)

            self.tot_loss += self.loss

            self.reconstruction_loss += self.net.reconstruction_loss
            self.kl_loss += self.net.kl_divergence_loss

            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            if self.iteration % self.exp_config.validation_frequency == 0:
                self.validate(data)

            if self.iteration % self.exp_config.logging_frequency == 0:
                logging.info('Iteration {} Loss {}'.format(self.iteration, self.loss))
                self._create_tensorboard_summary()
                self.tot_loss = 0
                self.kl_loss = 0
                self.reconstruction_loss = 0

            self.scheduler.step(self.loss)

        logging.info('Finished training.')

    def validate(self, data):
        self.net.eval()
        with torch.no_grad():
            logging.info('Validation for step {}'.format(self.iteration))

            logging.info('Checkpointing model.')
            self.save_model('validation_ckpt')

            ged_list = []
            dice_list = []
            ncc_list = []
            elbo_list = []
            kl_list = []
            recon_list = []

            time_ = time.time()

            validation_set_size = data.validation.images.shape[0]

            for ii in range(validation_set_size):

                s_gt_arr = data.validation.labels[ii, ...]

                # from HW to NCHW
                x_b = data.validation.images[ii, ...]
                patch = torch.tensor(x_b, dtype=torch.float32).to(self.device)
                val_patch = patch.unsqueeze(dim=0).unsqueeze(dim=1)

                s_b = s_gt_arr[:, :, np.random.choice(self.exp_config.annotator_range)]
                mask = torch.tensor(s_b, dtype=torch.float32).to(self.device)
                val_mask = mask.unsqueeze(dim=0).unsqueeze(dim=1)
                val_masks = torch.tensor(s_gt_arr, dtype=torch.float32).to(self.device)  # HWC
                val_masks = val_masks.transpose(0, 2).transpose(1, 2)  # CHW
                val_masks = val_masks.unsqueeze(dim=0)  # 1, annotations, H, W

                patch_arrangement = val_patch.repeat((self.exp_config.validation_samples, 1, 1, 1))

                mask_arrangement = val_mask.repeat((self.exp_config.validation_samples, 1, 1, 1))

                self.mask = mask_arrangement
                self.patch = patch_arrangement

                # training=True for constructing posterior as well
                self.net.forward(patch_arrangement, mask_arrangement, training=True) # sample N times
                self.val_loss = self.net.loss(mask_arrangement)
                elbo = self.val_loss
                kl = self.net.kl_divergence_loss
                recon = self.net.reconstruction_loss

                s_prediction_softmax = torch.softmax(self.net.sample(testing=True), dim=1)
                s_prediction_softmax_mean = torch.mean(s_prediction_softmax, axis=0)

                s_prediction_arrangement = torch.argmax(s_prediction_softmax, dim=1)

                ground_truth_arrangement = val_masks.transpose(0, 1)  # annotations, n_labels, H, W
                ged = utils.generalised_energy_distance(s_prediction_arrangement, ground_truth_arrangement,
                                                        nlabels=self.exp_config.n_classes - 1,
                                                        label_range=range(1, self.exp_config.n_classes))

                ground_truth_arrangement_one_hot = utils.convert_batch_to_onehot(ground_truth_arrangement, nlabels=self.exp_config.n_classes)
                ncc = utils.variance_ncc_dist(s_prediction_softmax, ground_truth_arrangement_one_hot)

                s_ = torch.argmax(s_prediction_softmax_mean, dim=0) # HW
                s = val_mask.view(val_mask.shape[-2], val_mask.shape[-1]) #HW

                # Write losses to list
                per_lbl_dice = []
                for lbl in range(self.exp_config.n_classes):
                    binary_pred = (s_ == lbl) * 1
                    binary_gt = (s == lbl) * 1

                    if torch.sum(binary_gt) == 0 and torch.sum(binary_pred) == 0:
                        per_lbl_dice.append(1.0)
                    elif torch.sum(binary_pred) > 0 and torch.sum(binary_gt) == 0 or torch.sum(binary_pred) == 0 and torch.sum(
                            binary_gt) > 0:
                        per_lbl_dice.append(0.0)
                    else:
                        per_lbl_dice.append(dc(binary_pred.detach().cpu().numpy(), binary_gt.detach().cpu().numpy()))

                dice_list.append(per_lbl_dice)
                elbo_list.append(elbo)
                kl_list.append(kl)
                recon_list.append(recon)

                ged_list.append(ged)
                ncc_list.append(ncc)

            dice_tensor = torch.tensor(dice_list)
            per_structure_dice = dice_tensor.mean(dim=0)

            elbo_tensor = torch.tensor(elbo_list)
            kl_tensor = torch.tensor(kl_list)
            recon_tensor = torch.tensor(recon_list)

            ged_tensor = torch.tensor(ged_list)
            ncc_tensor = torch.tensor(ncc_list)

            self.avg_dice = torch.mean(dice_tensor)
            self.foreground_dice = torch.mean(dice_tensor, dim=0)[1]
            self.val_elbo = torch.mean(elbo_tensor)
            self.val_recon_loss = torch.mean(recon_tensor)
            self.val_kl_loss = torch.mean(kl_tensor)

            self.avg_ged = torch.mean(ged_tensor)
            self.avg_ncc = torch.mean(ncc_tensor)

            logging.info(' - Foreground dice: %.4f' % torch.mean(self.foreground_dice))
            logging.info(' - Mean (neg.) ELBO: %.4f' % self.val_elbo)
            logging.info(' - Mean GED: %.4f' % self.avg_ged)
            logging.info(' - Mean NCC: %.4f' % self.avg_ncc)

            if torch.mean(per_structure_dice) >= self.best_dice:
                self.best_dice = torch.mean(per_structure_dice)
                logging.info('New best validation Dice! (%.3f)' % self.best_dice)
                self.save_model(savename='best_dice')
            if self.val_elbo <= self.best_loss:
                self.best_loss = self.val_elbo
                logging.info('New best validation loss! (%.3f)' % self.best_loss)
                self.save_model(savename='best_loss')
            if self.avg_ged <= self.best_ged:
                self.best_ged = self.avg_ged
                logging.info('New best GED score! (%.3f)' % self.best_ged)
                self.save_model(savename='best_ged')
            if self.avg_ncc >= self.best_ncc:
                self.best_ncc = self.avg_ncc
                logging.info('New best NCC score! (%.3f)' % self.best_ncc)
                self.save_model(savename='best_ncc')

            logging.info('Validation took {} seconds'.format(time.time()-time_))

        self.net.train()

    def _create_tensorboard_summary(self, end_of_epoch=False):
        self.net.eval()
        with torch.no_grad():
            # calculate the means since the last validation
            self.training_writer.add_scalar('Mean_loss', self.mean_loss_of_epoch/self.exp_config.validation_frequency, global_step=self.iteration)
            self.training_writer.add_scalar('KL_Divergence_loss', self.kl_loss/self.exp_config.validation_frequency, global_step=self.iteration)
            self.training_writer.add_scalar('Reconstruction_loss', self.reconstruction_loss/self.exp_config.validation_frequency, global_step=self.iteration)

            self.validation_writer.add_scalar('Dice_score_of_last_validation', self.foreground_dice, global_step=self.iteration)
            self.validation_writer.add_scalar('GED_score_of_last_validation', self.avg_ged, global_step=self.iteration)
            self.validation_writer.add_scalar('NCC_score_of_last_validation', self.avg_ncc, global_step=self.iteration)

            self.validation_writer.add_scalar('Mean_loss', self.val_elbo, global_step=self.iteration)
            self.validation_writer.add_scalar('KL_Divergence_loss', self.val_kl_loss, global_step=self.iteration)
            self.validation_writer.add_scalar('Reconstruction_loss', self.val_recon_loss, global_step=self.iteration)

            # plot images of current patch for summary
            sample = torch.softmax(self.net.sample(), dim=1)
            sample1 = torch.chunk(sample, 2, dim=1)[self.exp_config.n_classes-1]

            self.training_writer.add_image('Patch/GT/Sample',
                                          torch.cat([self.patch, self.mask.view(-1, 1, 128, 128), sample1], dim=2),
                                          global_step=self.iteration, dataformats='NCHW')
        self.net.train()

    def save_model(self, savename):
        model_name = self.exp_config.experiment_name + '_' + savename + '.pth'
        save_model_path = os.path.join(sys_config.project_root, 'models', model_name)
        torch.save(self.net.state_dict(), save_model_path)
        logging.info('saved model to .pth file in {}'.format(save_model_path))

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
    transform = None

    data = lidc_data(sys_config=sys_config, exp_config=exp_config)
    model.train(data)



    model.save_model()
