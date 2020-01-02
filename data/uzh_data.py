

import numpy as np
from scipy.io import loadmat
from data.batch_provider import BatchProvider, resize_batch

class uzh_data():

    def __init__(self, sys_config, exp_config):

        data = loadmat(sys_config.uzh_root)

        # Create the batch providers
        augmentation_options = exp_config.augmentation_options

        if hasattr(exp_config, 'resize_to'):
            resize_to = exp_config.resize_to
        else:
            resize_to = None

        if not hasattr(exp_config, 'annotator_range'):
            exp_config.annotator_range = range(exp_config.num_labels_per_subject)

        indices = list(range(data['X'].shape[0]))
        annotator_range = range(1)
        self.train = BatchProvider(data['X'][:-100], data['y'][:-100], indices[:-100],
                                   add_dummy_dimension=True,
                                   do_augmentations=True,
                                   augmentation_options=augmentation_options,
                                   num_labels_per_subject=1,
                                   annotator_range=annotator_range,
                                   resize_to=resize_to)
        self.validation = BatchProvider(data['X'][-100:-50], data['y'][-100:-50], indices[-100:-50],
                                        add_dummy_dimension=True,
                                        num_labels_per_subject=1,
                                        annotator_range=annotator_range,
                                        resize_to=resize_to)
        self.test = BatchProvider(data['X'][-50:], data['y'][-50:], indices[-50:],
                                  add_dummy_dimension=True,
                                  num_labels_per_subject=1,
                                  annotator_range=annotator_range,
                                  resize_to=resize_to)

        self.test.images = resize_batch(data['X'][-50:], target_size=resize_to)
        self.test.labels = resize_batch(data['y'][-50:], target_size=resize_to).reshape((-1,
                                                                                     resize_to[0],resize_to[1],
                                                                                     exp_config.num_labels_per_subject))

        self.validation.images = resize_batch(data['X'][-100:-50], target_size=resize_to)
        self.validation.labels = resize_batch(data['y'][-100:-50], target_size=resize_to).reshape((-1,
                                                                                               resize_to[0],resize_to[1],
                                                                                               exp_config.num_labels_per_subject))


if __name__ == '__main__':

    # If the program is called as main, perform some debugging operations
    from models.experiments import phiseg_uzh_rev_7_5_192 as exp_config
    from config import local_config as sys_config
    data = uzh_data(sys_config, exp_config)

