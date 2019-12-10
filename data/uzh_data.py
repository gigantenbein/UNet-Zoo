

import numpy as np
from scipy.io import loadmat
from data.batch_provider import BatchProvider

class uzh_data():

    def __init__(self, sys_config, exp_config):

        data = loadmat(sys_config.uzh_root)

        # Create the batch providers
        augmentation_options = exp_config.augmentation_options
        indices = list(range(data['X'].shape[0]))
        annotator_range = range(1)
        self.train = BatchProvider(data['X'][:-100], data['y'][:-100], indices[:-100],
                                   add_dummy_dimension=True,
                                   do_augmentations=True,
                                   augmentation_options=augmentation_options,
                                   num_labels_per_subject=1,
                                   annotator_range=annotator_range)
        self.validation = BatchProvider(data['X'][-100:-50], data['y'][-100:-50], indices[-100:-50],
                                        add_dummy_dimension=True,
                                        num_labels_per_subject=1,
                                        annotator_range=annotator_range)
        self.test = BatchProvider(data['X'][-50:], data['y'][-50:], indices[-50:],
                                  add_dummy_dimension=True,
                                  num_labels_per_subject=1,
                                  annotator_range=annotator_range)

        self.test.images = data['X'][-50:]
        self.test.labels = data['y'][-50:]

        self.validation.images = data['X'][-100:-50]
        self.validation.labels = data['y'][-100:-50]


if __name__ == '__main__':

    # If the program is called as main, perform some debugging operations
    from models.experiments import phiseg as exp_config
    data = uzh_data(exp_config)

