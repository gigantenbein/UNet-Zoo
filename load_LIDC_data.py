import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def load_data_into_loader(sys_config, name):
    location = os.path.join(sys_config.data_root, name)
    dataset = LIDC_IDRI(dataset_location=location)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.shuffle(indices)

    train_indices, test_indices, val_indices = indices[2*split:], indices[2*split:3*split], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=5, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
    validation_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)
    print("Number of training/test/validation patches:", (len(train_indices),len(test_indices), len(val_indices)))

    return train_loader, test_loader, validation_loader


def create_pickle_data_with_n_samples(sys_config, n=100):
    dataset = LIDC_IDRI(dataset_location=sys_config.data_root)

    print('Create pickle file with {} samples'.format(n))
    data_list = []
    for i in range(n):
        values = {}
        values['image'] = dataset.images[i]
        values['masks'] = dataset.labels[i]
        values['series_uid'] = dataset.series_uid[i]

        data_list.append(values)

    data_list = dict(zip(list(range(n)), data_list))

    with open('/Users/marcgantenbein/scratch/data/size1000/lidc_data_length_{}.pickle'.format(n), 'wb') as file:
        pickle.dump(data_list, file)


class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []

    def __init__(self, dataset_location, transform=None):
        self.transform = transform
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        
        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)

        #Randomly select one of the four labels for this image
        label = self.labels[index][random.randint(0,3)].astype(float)
        labels_ = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)

        series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        labels_ = torch.from_numpy(np.array(labels_))

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        labels_ = labels_.type(torch.FloatTensor)

        return image, label, series_uid, labels_

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
