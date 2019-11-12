import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Python bundle packages
import os
import logging
import shutil
from importlib.machinery import SourceFileLoader
import pickle

# own files
from load_LIDC_data import LIDC_IDRI
import argparse
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class UNetModel:
    def __init__(self, exp_config):

        self.net = exp_config.model(input_channels=exp_config.input_channels,
                                    num_classes=exp_config.n_classes,
                                    num_filters=exp_config.filter_channels,
                                    latent_dim=exp_config.latent_levels,
                                    no_convs_fcomb=exp_config.no_convs_fcomb,
                                    beta=exp_config.beta,
                                    reversible=exp_config.use_reversible
                                    )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=0)

        self.epochs = exp_config.epochs_to_train

    def train(self, train_loader):
        self.net.train()
        logging.info('Starting training.')
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        for epoch in range(self.epochs):
            for step, (patch, mask, _, __) in enumerate(train_loader):
                patch = patch.to(self.device)
                mask = mask.to(self.device)
                mask = torch.unsqueeze(mask, 1)

                self.net.forward(patch, mask, training=True)
                loss = self.net.loss(mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                if step % 100 == 0:
                    logging.info('Epoch {} Step {} Loss {}'.format(epoch, step, loss))
                    logging.info('Epoch: {} Number of processed patches: {}'.format(epoch, step))
            logging.info('Finished epoch {}'.format(epoch))
        logging.info('Finished training.')

    def validate(self):
        pass





def load_data_into_loader(sys_config):
    dataset = LIDC_IDRI(dataset_location=sys_config.data_root)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=5, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
    print("Number of training/test patches:", (len(train_indices),len(test_indices)))

    return train_loader, test_loader


def load_dummy_dataset():
    with open(os.path.join(sys_config.data_root, 'dummy/dummy.pickle'), 'rb') as handle:
        dummy = pickle.load(handle)
        return dummy


def dummy_train():
    """Feed the model with one image and one label"""
    dataset = load_dummy_dataset()

    patch = dataset[0].view(1, 1, 128, 128).to(device)
    mask = dataset[1].view(1, 1, 128, 128).to(device)


    print(net)
    net.forward(patch, mask, training=True)
    elbo = net.elbo(mask)
    reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(
        net.fcomb.layers)
    loss = -elbo + 1e-5 * reg_loss
    loss = -elbo
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('step')


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
    else:
        import config.system as sys_config

    logging.info('Running experiment with script: {}'.format(config_file))

    exp_config = SourceFileLoader(config_module, config_file).load_module()

    log_dir = os.path.join(sys_config.log_root, exp_config.log_dir_name, exp_config.experiment_name)

    utils.makefolder(log_dir)

    shutil.copy(exp_config.__file__, log_dir)
    logging.info('!!!! Copied exp_config file to experiment folder !!!!')

    writer = SummaryWriter()

    logging.info('**************************************************************')
    logging.info(' *** Running Experiment: %s', exp_config.experiment_name)
    logging.info('**************************************************************')

    model = UNetModel(exp_config)

    if args.dummy == 'dummy':
        dummy_train()
    else:
        train_loader, test_loader = load_data_into_loader(sys_config)
        model.train(train_loader)

    logging.info('Finished training the model.')

    model_name = exp_config.experiment_name + '.pth'
    save_model_path = os.path.join(sys_config.project_root, 'models', model_name)
    torch.save(net.state_dict(), save_model_path)
    logging.info('saved model to .pth file in {}'.format(save_model_path))

