# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import numpy as np

from data import uzh_prostate_data_loader
from data.batch_provider import BatchProvider


class uzh_prostate_data():

    def __init__(self, sys_config, exp_config):

        data = uzh_prostate_data_loader.load_and_maybe_process_data(
            input_image_folder=exp_config.input_image_folder,
            input_mask_folder=exp_config.input_mask_folder,
            preprocessing_folder=exp_config.preproc_folder,
            size=exp_config.image_size[1:3],
            target_resolution=(0.6, 0.6),
            force_overwrite=False,
        )

        self.data = data

        label_name = 'masks'

        # the following are HDF5 datasets, not numpy arrays
        images_train = data['images_train']
        labels_train = data['%s_train' % label_name]

        images_test = data['images_test']
        labels_test = data['%s_test' % label_name]

        images_val = data['images_validation']
        labels_val = data['%s_validation' % label_name]

        # Extract the number of training and testing points
        N_train = images_train.shape[0]
        N_test = images_test.shape[0]
        N_val = images_val.shape[0]

        # Create a shuffled range of indices for both training and testing data
        train_indices = np.arange(N_train)
        test_indices = np.arange(N_test)
        val_indices = np.arange(N_val)

        # Create the batch providers
        augmentation_options = exp_config.augmentation_options

        # Backwards compatibility, TODO remove for final version
        # if not hasattr(exp_config, 'annotator_range'):
        #     exp_config.annotator_range = range(exp_config.num_labels_per_subject)

        self.train = BatchProvider(images_train, labels_train, train_indices,
                                   add_dummy_dimension=True,
                                   do_augmentations=True,
                                   augmentation_options=augmentation_options,
                                   num_labels_per_subject=exp_config.num_labels_per_subject,
                                   annotator_range=exp_config.annotator_range
        )
        self.validation = BatchProvider(images_val, labels_val, val_indices,
                                        add_dummy_dimension=True,
                                        num_labels_per_subject=exp_config.num_labels_per_subject,
                                        annotator_range=exp_config.annotator_range)
        self.test = BatchProvider(images_test, labels_test, test_indices,
                                  add_dummy_dimension=True,
                                  num_labels_per_subject=exp_config.num_labels_per_subject,
                                  annotator_range=exp_config.annotator_range)

        self.test.images = images_test
        self.test.labels = labels_test

        self.validation.images = images_val
        self.validation.labels = labels_val


if __name__ == '__main__':

    # If the program is called as main, perform some debugging operations
    from phiseg.experiments.LIDC import phiseg_7_5 as exp_config

    data = uzh_prostate_data(exp_config)

    print('DEBUGGING OUTPUT')
    print('training')
    for ii in range(2):
        X_tr, Y_tr = data.train.next_batch(10)
        print(np.mean(X_tr))
        print(Y_tr.shape)
        print('--')

    print('test')
    for ii in range(2):
        X_te, Y_te = data.test.next_batch(10)
        print(np.mean(X_te))
        print(Y_te.shape)
        print('--')

    print('validation')
    for ii in range(2):
        X_va, Y_va = data.validation.next_batch(10)
        print(np.mean(X_va))
        print(Y_va.shape)
        print('--')

    for ii, bb in enumerate(data.train.iterate_batches(1)):

        x, y = bb

        import matplotlib.pyplot as plt
        plt.imshow(np.squeeze(y))
        plt.show()

        if ii > 3:
            break
