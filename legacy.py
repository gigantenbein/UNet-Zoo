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
        basic_logger.info('Starting training.')

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
                    basic_logger.info('Epoch {} Step {} Loss {}'.format(self.epoch, self.step, self.loss))
                    basic_logger.info('Epoch: {} Number of processed patches: {}'.format(self.epoch, self.step))
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

            basic_logger.info('Finished epoch {}'.format(self.epoch))
            print('Finished epoch {}'.format(self.epoch))
        basic_logger.info('Finished training.')

def validate(self, validation_loader):
    self.net.eval()
    with torch.no_grad():
        basic_logger.info('Validation for step {}'.format(self.iteration))

        ged_list = []
        dice_list = []
        ncc_list = []
        elbo_list = []
        kl_list = []
        recon_list = []

        time_ = time.time()

        for val_step, (val_patch, val_mask, _, val_masks) in enumerate(validation_loader):
            val_patch = val_patch.to(self.device)
            val_mask = val_mask.to(self.device)
            val_masks = val_masks.to(self.device)

            patch_arrangement = val_patch.repeat((self.exp_config.validation_samples, 1, 1, 1))

            mask_arrangement = val_mask.repeat((self.exp_config.validation_samples, 1, 1, 1))

            self.net.forward(patch_arrangement, mask_arrangement, training=True)  # sample N times
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

            ground_truth_arrangement_one_hot = utils.convert_to_onehot(ground_truth_arrangement,
                                                                       nlabels=self.exp_config.n_classes)
            ncc = utils.variance_ncc_dist(s_prediction_softmax, ground_truth_arrangement_one_hot)

            s_ = torch.argmax(s_prediction_softmax_mean, dim=0)  # HW
            s = val_mask.view(val_mask.shape[-2], val_mask.shape[-1])  # HW

            # Write losses to list
            per_lbl_dice = []
            for lbl in range(self.exp_config.n_classes):
                binary_pred = (s_ == lbl) * 1
                binary_gt = (s == lbl) * 1

                if torch.sum(binary_gt) == 0 and torch.sum(binary_pred) == 0:
                    per_lbl_dice.append(1.0)
                elif torch.sum(binary_pred) > 0 and torch.sum(binary_gt) == 0 or torch.sum(
                        binary_pred) == 0 and torch.sum(
                        binary_gt) > 0:
                    per_lbl_dice.append(0.0)
                else:
                    per_lbl_dice.append(
                        dc(binary_pred.detach().cpu().numpy(), binary_gt.detach().cpu().numpy()))

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

        basic_logger.info(' - Mean dice: %.4f' % torch.mean(per_structure_dice))
        basic_logger.info(' - Mean (neg.) ELBO: %.4f' % self.val_elbo)
        basic_logger.info(' - Mean GED: %.4f' % self.avg_ged)
        basic_logger.info(' - Mean NCC: %.4f' % self.avg_ncc)

        basic_logger.info('Validation took {} seconds'.format(time.time() - time_))

    self.net.train()


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

def test_quantitative(model_path, exp_config, sys_config, do_plots=False):
    n_samples = 50
    model_selection = 'best_ged'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get Data
    net = exp_config.model(input_channels=exp_config.input_channels,
                           num_classes=exp_config.n_classes,
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
                basic_logger.info("Progress: %d" % ii)
                print("Progress: {} GED: {}".format(ii, ged))

            net.forward(patch, mask=mask, training=False)
            sample = torch.nn.functional.softmax(net.sample(testing=True))
            ground_truth_labels = masks.view(-1,1,128,128)

            # Generalized energy distance
            ged = utils.generalised_energy_distance(sample, ground_truth_labels, 4, label_range=range(1, 5))
            ged_list.append(ged)

            # Dice coefficient
            dice = dc(sample.view(-1, 128, 128).detach().numpy(), mask.view(-1, 128, 128).detach().numpy())
            dice_list.append(dice)

            # Normalised Cross correlation
            ncc = utils.variance_ncc_dist(sample, ground_truth_labels)
            ncc_list.append(ncc)



    ged_arr = np.asarray(ged_list)
    ncc_arr = np.asarray(ncc_list)
    dice_arr = np.asarray(dice_list)

    print('-- GED: --')
    print(np.mean(ged_arr))
    print(np.std(ged_arr))

    print(' -- NCC: --')
    print(np.mean(ncc_arr))
    print(np.std(ncc_arr))

    print(' -- Dice: --')
    print(np.mean(dice_arr))
    print(np.std(dice_arr))

    basic_logger.info('-- GED: --')
    basic_logger.info(np.mean(ged_arr))
    basic_logger.info(np.std(ged_arr))

    basic_logger.info('-- NCC: --')
    basic_logger.info(np.mean(ncc_arr))
    basic_logger.info(np.std(ncc_arr))

    basic_logger.info('-- Dice: --')
    basic_logger.info(np.mean(dice_arr))
    basic_logger.info(np.std(dice_arr))

   # np.savez(os.path.join(model_path, 'ged%s_%s.npz' % (str(n_samples), model_selection)), ged_arr)
   # np.savez(os.path.join(model_path, 'ncc%s_%s.npz' % (str(n_samples), model_selection)), ncc_arr)


def test_segmentation(exp_config, sys_config, amount_of_tests=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get Data
    net = exp_config.model(input_channels=exp_config.input_channels,
                           num_classes=exp_config.n_classes,
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

    _, data, val_data = load_data_into_loader(sys_config,'')

    with torch.no_grad():
        for ii, (patch, mask, _, masks) in enumerate(data):

            if ii > amount_of_tests:
                break
            if ii % 10 == 0:
                basic_logger.info("Progress: %d" % ii)
                print("Progress: {}".format(ii))

            net.forward(patch, mask, training=False)
            sample = torch.sigmoid(net.sample())

           # sample = torch.where(sample > 0.5, 1, 0)
            n = min(patch.size(0), 8)
            comparison = torch.cat([patch[:n],
                                     masks.view(-1, 1, 128, 128),
                                     sample[:n]])
            #comparison = sample.view(-1, 1, 128, 128)
            save_image(comparison.cpu(),
                       'segmentation/' + exp_config.experiment_name + '/comp_' + str(ii) + '.png', nrow=n)


def generate_images():
    from torchvision.utils import save_image
    for ii in range(100):
        save_patch = torch.tensor(data.validation.images[ii, ...], dtype=torch.float)
        save_labels = torch.tensor(data.validation.labels[ii, ...], dtype=torch.float) # HWC

        save_labels = save_labels.transpose(0,2).transpose(1,2) #CHW
        save_labels = save_labels.view(-1,1,128,128)
        save_patch = save_patch.view(-1,1,128,128)

        save = torch.cat([save_patch, save_labels], dim=0)

        save_image(save, 'test{}.png'.format(ii), pad_value=1, scale_each=True, normalize=True)