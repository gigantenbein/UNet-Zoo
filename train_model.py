import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
from torchvision.utils import save_image
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from no_new_reversible_net import NoNewReversible

def test(epoch):
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


if __name__ == '__main__':
    writer = SummaryWriter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LIDC_IDRI(dataset_location = '/Users/marcgantenbein/scratch/data/')
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


    # for epoch in range(epochs):
    #     for step, (patch, mask, _) in enumerate(train_loader):
    #         patch = patch.to(device)
    #         mask = mask.to(device)
    #         mask = torch.unsqueeze(mask,1)
    #         prediction = net.forward(patch)
    #         CEloss = nn.CrossEntropyLoss()
    #         loss = CEloss(
    #             prediction,
    #             mask.view(-1, 128, 128).long()
    #         )
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=2,
                            no_convs_fcomb=4, beta=10.0, reversible=True)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
    epochs = 10

    net.train()
    test(1)
    for epoch in range(epochs):
        for step, (patch, mask, _) in enumerate(train_loader):
            patch = patch.to(device)
            mask = mask.to(device)
            mask = torch.unsqueeze(mask, 1)
            # if step == 0:
            #     writer.add_graph(
            #         ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=2,
            #                           no_convs_fcomb=4, beta=10.0, reversible=True), [patch, mask], verbose=True)
            #     writer.close()

            net.forward(patch, mask, training=True)
            elbo = net.elbo(mask)
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(
                net.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('step')


    torch.save(net.state_dict(), './models/prob_unet.pth')


