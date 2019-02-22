from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
import glob
import os
from base_data_loader import BaseDataLoader


def get_ssl_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    name_config = config['data_loader']['type']
    arg_config = config['data_loader']['args']
    train_config = {
        'root': '../data/%s' % (name_config),
        'train': True,
        'transform': transform,
        "download": True,
    }
    dev_config = {
        'root': '../data/%s' % (name_config),
        'train': 'False',
        'transform': transform,
        "download": True,
    }

    training_set = getattr(datasets, name_config)(**train_config)
    dev_set = getattr(datasets, name_config)(**dev_config)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(arg_config['size_labeled_data'] / 10)]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size',
          len(dev_set))

    labeled_loader = BaseDataLoader(training_set, labeled_indices, arg_config['train_batch_size'])
    unlabeled_loader = BaseDataLoader(training_set, unlabeled_indices, arg_config['train_batch_size'])
    dev_loader = BaseDataLoader(dev_set, np.arange(len(dev_set)), arg_config['dev_batch_size'])

    return labeled_loader, unlabeled_loader, dev_loader


class raw_dataset(Dataset):

    def __init__(self, image_list, label_list, phase, transform=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])

        label = self.label_list[idx]

        sample = [image, label]

        if self.transform:
            sample[0] = self.transform(sample[0], self.phase)

        return sample


def image_transform(image, phase):
    if phase == 'train':
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)

    # Resize
    image = transforms.Resize((32, 32))(image)
    # Transform to tensor
    image = transforms.ToTensor()(image)
    # Normalize
    image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

    return image


def raw_dataloader(image_list, label_list, phase, batch_size):
    dataset = raw_dataset(image_list, label_list, phase, transform=image_transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=(phase == 'train'), num_workers=8, drop_last=False)

    return dataloader


def get_patch_loaders(config):
    arg_config = config['data_loader']['args']
    dataset_root = arg_config['dataset_root']
    train_root = os.path.join(dataset_root, 'train')
    test_root = os.path.join(dataset_root, 'test')

    train_pos = glob.glob(train_root + '/pos/*.png')
    train_neg = glob.glob(train_root + '/neg/*.png')

    test_pos = glob.glob(test_root + '/pos/*.png')
    test_neg = glob.glob(test_root + '/neg/*.png')

    train_image = np.array(train_pos + train_neg)
    train_label = np.array([1] * len(train_pos) + [0] * len(train_neg))

    test_image = np.array(test_pos + test_neg)
    test_label = np.array([1] * len(test_pos) + [0] * len(test_neg))

    print(train_root)

    training_set = raw_dataloader(train_image, train_label, 'train', arg_config['train_batch_size'])
    dev_set = raw_dataloader(test_image, test_label, 'test', arg_config['dev_batch_size'])

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(arg_config['size_labeled_data'] / 10)]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size',
          len(dev_set))

    labeled_loader = BaseDataLoader(training_set, labeled_indices, arg_config['train_batch_size'])
    unlabeled_loader = BaseDataLoader(training_set, unlabeled_indices, arg_config['train_batch_size'])
    dev_loader = BaseDataLoader(dev_set, np.arange(len(dev_set)), arg_config['dev_batch_size'])

    return labeled_loader, unlabeled_loader, dev_loader


if __name__ == '__main__':
    import json

    a, b, c = get_patch_loaders(json.load(open('../config/config_prostate.json')))
    print(len(a))
