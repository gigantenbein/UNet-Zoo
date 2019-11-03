import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.phiseg import PHISeg

# Python bundle packages
import os
import logging
import shutil
from importlib.machinery import SourceFileLoader
import pickle

# own files
from load_LIDC_data import LIDC_IDRI
from utils import l2_regularisation
import argparse
import utils
from models.probabilistic_unet import ProbabilisticUnet

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


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
    test_loader = DataLoader(dataset, batch_size=5, sampler=test_sampler)
    print("Number of training/test patches:", (len(train_indices),len(test_indices)))

    return train_loader, test_loader


def train(train_loader, epochs):
    net.train()
    for epoch in range(epochs):
        for step, (patch, mask, _) in enumerate(train_loader):
            patch = patch.to(device)
            mask = mask.to(device)
            mask = torch.unsqueeze(mask, 1)

            net.forward(patch, mask, training=True)
            elbo = net.elbo(mask)
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(
                net.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('step')


def test(test_loader):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for step, (patch, mask, _) in enumerate(test_loader):
            patch = patch.to(device)
            mask = mask.to(device)

            net.forward(patch, mask, training=False)

            prediction = net.unet_features

            CEloss = nn.CrossEntropyLoss()
            test_loss = CEloss(
                prediction,
                mask.view(-1, 128, 128).long(),
            ).item()

            n = min(patch.size(0), 8)
            comparison = torch.cat([patch[:n],
                                    mask.view(-1, 1, 128, 128)[:n],
                                   prediction.view(-1, 1, 128, 128)[:n]])
            save_image(comparison.cpu(),
                       'segmentation/comp_' + str(step) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


def load_dummy_dataset():
    with open(os.path.join(sys_config.data_root, 'dummy/dummy.pickle'), 'rb') as handle:
        dummy = pickle.load(handle)
        return dummy


def dummy_train():
    """Feed the model with one image and one label"""
    dataset = load_dummy_dataset()

    patch = dataset[0].view(1, 1, 128, 128).to(device)
    mask = dataset[1].view(1, 1, 128, 128).to(device)

    net.forward(patch, mask, training=True)
    elbo = net.elbo(mask)
    reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(
        net.fcomb.layers)
    loss = -elbo + 1e-5 * reg_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_graph(net, [patch, mask])
    writer.close()
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = exp_config.model(input_channels=exp_config.input_channels,
                           num_classes=1,
                           num_filters=exp_config.filter_channels,
                           latent_dim=2,
                           no_convs_fcomb=4,
                           beta=10.0,
                           reversible=exp_config.use_reversible
                           )



    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)

    epochs = 100

    if args.dummy == 'dummy':
        dummy_train()
    else:
        train_loader, test_loader = load_data_into_loader(sys_config)
        train(train_loader, epochs)

    logging.info('Finished training the model.')

    model_name = exp_config.experiment_name + '.pth'
    save_model_path = os.path.join(sys_config.project_root, 'models', model_name)
    torch.save(net.state_dict(), save_model_path)
    logging.info('saved model to .pth file in {}'.format(save_model_path))

