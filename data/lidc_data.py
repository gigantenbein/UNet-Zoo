# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import numpy as np
from data import lidc_data_loader
from data.batch_provider import BatchProvider

class lidc_data():

    def __init__(self, sys_config,exp_config):

        data = lidc_data_loader.load_and_maybe_process_data(
            input_file=sys_config.data_root,
            preprocessing_folder=sys_config.preproc_folder,
            force_overwrite=False,
        )

        self.data = data

        # Extract the number of training and testing points
        indices = {}
        for tt in data:
            N = data[tt]['images'].shape[0]
            indices[tt] = np.arange(N)

        # Create the batch providers
        augmentation_options = exp_config.augmentation_options

        # Backwards compatibility, TODO remove for final version
        if not hasattr(exp_config, 'annotator_range'):
            exp_config.annotator_range = range(exp_config.num_labels_per_subject)

        self.train = BatchProvider(data['train']['images'], data['train']['labels'], indices['train'],
                                   add_dummy_dimension=True,
                                   do_augmentations=True,
                                   augmentation_options=augmentation_options,
                                   num_labels_per_subject=exp_config.num_labels_per_subject,
                                   annotator_range=exp_config.annotator_range)
        self.validation = BatchProvider(data['val']['images'], data['val']['labels'], indices['val'],
                                        add_dummy_dimension=True,
                                        num_labels_per_subject=exp_config.num_labels_per_subject,
                                        annotator_range=exp_config.annotator_range)
        self.test = BatchProvider(data['test']['images'], data['test']['labels'], indices['test'],
                                  add_dummy_dimension=True,
                                  num_labels_per_subject=exp_config.num_labels_per_subject,
                                  annotator_range=exp_config.annotator_range)

        self.test.images = data['test']['images']
        self.test.labels = data['test']['labels']

        self.validation.images = data['val']['images']
        self.validation.labels = data['val']['labels']