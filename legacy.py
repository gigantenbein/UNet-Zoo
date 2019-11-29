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

    def train(self, train_loader, validation_loader):
        self.net.train()
        logging.info('Starting training.')

        for self.epoch in range(self.epochs):
            self.validate(validation_loader)
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

                self.tot_loss += self.loss
                self.loss_list.append(self.loss)

                self.reconstruction_loss_list.append(self.net.reconstruction_loss)
                self.kl_loss_list.append(self.net.kl_divergence_loss)

                assert math.isnan(self.loss) == False

                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                print('Epoch {} Step {} Loss {}'.format(self.epoch, self.step, self.loss))
                if self.step % exp_config.logging_frequency == 0:
                    logging.info('Epoch {} Step {} Loss {}'.format(self.epoch, self.step, self.loss))
                    logging.info('Epoch: {} Number of processed patches: {}'.format(self.epoch, self.step))
                    print('Epoch {} Step {} Loss {}'.format(self.epoch, self.step, self.loss))
                    print('Epoch: {} Number of processed patches: {}'.format(self.epoch, self.step))
                    self._create_tensorboard_summary()
                if self.step % exp_config.validation_frequency == 0:
                    self._create_tensorboard_summary()
                    self.validate(validation_loader)
                self.scheduler.step(self.loss)

            self.mean_loss_of_epoch = sum(self.loss_list)/len(self.loss_list)

            self.kl_loss = sum(self.kl_loss_list)/len(self.kl_loss_list)
            self.reconstruction_loss = sum(self.reconstruction_loss_list)/len(self.reconstruction_loss_list)
            self._create_tensorboard_summary()
            self.validate(validation_loader)
            self._create_tensorboard_summary(end_of_epoch=True)

            self.tot_loss = 0
            self.kl_loss = 0
            self.reconstruction_loss = 0
            self.val_loss = 0
            self.loss_list = []
            self.reconstruction_loss_list = []
            self.kl_loss_list = []

            logging.info('Finished epoch {}'.format(self.epoch))
            print('Finished epoch {}'.format(self.epoch))
        logging.info('Finished training.')

if __name__ == '__main__':
    if args.dummy == 'dummy':
        train_loader, test_loader, validation_loader = load_data_into_loader(
            sys_config, 'size1000/', batch_size=exp_config.batch_size, transform=transform)
        utils.makefolder(os.path.join(sys_config.project_root, 'segmentation/', exp_config.experiment_name))
        model.train(train_loader, validation_loader)
    else:
        # train_loader, test_loader, validation_loader = load_data_into_loader(
        #     sys_config, '', batch_size=exp_config.batch_size, transform=transform)
        # utils.makefolder(os.path.join(sys_config.project_root, 'segmentation/', exp_config.experiment_name))
        # model.train(train_loader, validation_loader)
        # model.save_model()
        model.train(data)