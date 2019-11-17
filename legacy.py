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