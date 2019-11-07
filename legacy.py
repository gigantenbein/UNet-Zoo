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