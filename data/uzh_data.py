

import numpy as np
from scipy.io import loadmat
from data.batch_provider import BatchProvider, resize_batch
import os
import h5py


def load_uzh_data(input_file, output_file):
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    hdf5_file = h5py.File(output_file, "w")
    max_bytes = 2 ** 31 - 1

    data = {}
    file_path = os.fsdecode(input_file)
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    new_data = pickle.loads(bytes_in)
    data.update(new_data)

    series_uid = []

    for key, value in data.items():
        series_uid.append(value['series_uid'])

    unique_subjects = np.unique(series_uid)

    split_ids = {}
    train_and_val_ids, split_ids['test'] = train_test_split(unique_subjects, test_size=0.2)
    split_ids['train'], split_ids['val'] = train_test_split(train_and_val_ids, test_size=0.2)

    images = {}
    labels = {}
    uids = {}
    groups = {}

    for tt in ['train', 'test', 'val']:
        images[tt] = []
        labels[tt] = []
        uids[tt] = []
        groups[tt] = hdf5_file.create_group(tt)

    for key, value in data.items():

        s_id = value['series_uid']

        tt = find_subset_for_id(split_ids, s_id)

        images[tt].append(value['image'].astype(float)-0.5)

        lbl = np.asarray(value['masks'])  # this will be of shape 4 x 128 x 128
        lbl = lbl.transpose((1,2,0))

        labels[tt].append(lbl)
        uids[tt].append(hash(s_id))  # Checked manually that there are no collisions

    for tt in ['test', 'train', 'val']:

        groups[tt].create_dataset('uids', data=np.asarray(uids[tt], dtype=np.int))
        groups[tt].create_dataset('labels', data=np.asarray(labels[tt], dtype=np.uint8))
        groups[tt].create_dataset('images', data=np.asarray(images[tt], dtype=np.float))

    hdf5_file.close()


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

