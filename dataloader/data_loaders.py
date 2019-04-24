import glob
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10

from .base_data_loader import BaseParallelBaseDataLoader


class DataLoader(object):

    def __init__(self, config, raw_loader, indices, batch_size):
        self.images, self.labels = [], []
        for idx in indices:
            image, label = raw_loader[idx]
            self.images.append(image)
            self.labels.append(label)

        self.images = torch.stack(self.images, 0)
        self.labels = torch.from_numpy(np.array(self.labels, dtype=np.int64)).squeeze()

        if config['data_loader']['type'] == 'MNIST':
            self.images = self.images.view(self.images.size(0), -1)

        self.batch_size = batch_size

        self.unlimit_gen = self.generator(True)
        self.len = len(indices)

    def generator(self, inf=False):
        while True:
            indices = np.arange(self.images.size(0))
            np.random.shuffle(indices)
            indices = torch.from_numpy(indices)
            for start in range(0, indices.size(0), self.batch_size):
                end = min(start + self.batch_size, indices.size(0))
                ret_images, ret_labels = self.images[indices[start: end]], self.labels[indices[start: end]]
                yield ret_images, ret_labels
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return self.len


def get_mnist_loaders(config):
    config_dataloader = config['data_loader']['args']
    transform = transforms.Compose([transforms.ToTensor()])
    training_set = MNIST('../data/MNIST', train=True, download=True, transform=transform)
    dev_set = MNIST('../data/MNIST', train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(config_dataloader['size_labeled_data'] / 10)]] = True
    # here unlabeled dataset includes labeled data, use as many data as possible for gan training
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'test size',
          len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config_dataloader['train_batch_size'])
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config_dataloader['train_batch_size'])
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config_dataloader['train_batch_size'])
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config_dataloader['dev_batch_size'])

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader


def get_svhn_loaders(config):
    config_dataloader = config['data_loader']['args']
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = SVHN('../data/SVHN', split='train', download=True, transform=transform)
    dev_set = SVHN('../data/SVHN', split='test', download=True, transform=transform)

    def preprocess(data_set):
        for i in range(len(data_set)):
            if data_set.labels[i] == 10:
                data_set.labels[i] = 0
    preprocess(training_set)
    preprocess(dev_set)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(config_dataloader['size_labeled_data'] / 10)]] = True
    # here unlabeled dataset includes labeled data, use as many data as possible for gan training
    labeled_indices, unlabeled_indices = indices[mask], indices
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'test size',
          len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config_dataloader['train_batch_size'])
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config_dataloader['train_batch_size'])
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config_dataloader['train_batch_size'])
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config_dataloader['dev_batch_size'])

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader


def get_cifar_loaders(config):
    config_dataloader = config['data_loader']['args']
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = CIFAR10('../data/CIFAR', train=True, download=True, transform=transform)
    dev_set = CIFAR10('../data/CIFAR', train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(config_dataloader['size_labeled_data'] / 10)]] = True
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    labeled_indices, unlabeled_indices = indices[mask], indices
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size',
          len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config_dataloader['train_batch_size'])
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config_dataloader['train_batch_size'])
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config_dataloader['train_batch_size'])
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config_dataloader['dev_batch_size'])

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader


def get_ssl_loaders(config):
    name_config = config['data_loader']['type']

    if name_config == 'prostate':
        return get_parallel_patch_loaders(config)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    arg_config = config['data_loader']['args']

    if name_config == 'MNIST':
        train_config = {
            'root': '../data/%s' % (name_config),
            'train': True,
            'transform': transform,
            "download": True,
        }
        dev_config = {
            'root': '../data/%s' % (name_config),
            'train': False,
            'transform': transform,
            "download": True,
        }

    if name_config == 'SVHN':
        train_config = {
            'root': '../data/%s' % (name_config),
            'split': 'train',
            'transform': transform,
            "download": True,
        }
        dev_config = {
            'root': '../data/%s' % (name_config),
            'split': 'test',
            'transform': transform,
            "download": True,
        }

    if name_config == 'CIFAR10':
        train_config = {
            'root': '../data/%s' % (name_config),
            'train': True,
            'transform': transform,
            "download": True,
        }
        dev_config = {
            'root': '../data/%s' % (name_config),
            'train': False,
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

    labeled_set = deepcopy(training_set)
    unlabeled_set = deepcopy(training_set)
    labeled_set.train_labels = labeled_set.train_labels[labeled_indices]
    labeled_set.train_data = labeled_set.train_data[labeled_indices]
    unlabeled_set.train_labels = unlabeled_set.train_labels[unlabeled_indices]
    unlabeled_set.train_data = unlabeled_set.train_data[unlabeled_indices]

    labeled_loader = BaseParallelBaseDataLoader(config, labeled_set, 'train')
    unlabeled_loader = BaseParallelBaseDataLoader(config, unlabeled_set, 'train')
    dev_loader = BaseParallelBaseDataLoader(config, dev_set, 'test')

    return labeled_loader, unlabeled_loader, dev_loader


class raw_dataset(Dataset):

    def __init__(self, image_list, label_list, phase):
        self.image_list = image_list
        self.label_list = label_list
        self.phase = phase

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        image = self._image_transform(image, self.phase)

        label = self.label_list[idx]

        sample = [image, label]

        return sample

    def _image_transform(self, image, phase):
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


def get_parallel_patch_loaders(config):
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

    indices = np.arange(len(train_image))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([train_label[i] for i in indices], dtype=np.int64)

    num_label = config['model']['num_label']

    for i in range(num_label):
        mask[np.where(labels == i)[0][: int(arg_config['size_labeled_data'] / num_label)]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]

    labeled_set = raw_dataset(train_image[labeled_indices], train_label[labeled_indices], 'train')
    unlabeled_set = raw_dataset(train_image[unlabeled_indices], train_label[unlabeled_indices], 'train')
    dev_set = raw_dataset(test_image, test_label, 'test')

    labeled_loader = BaseParallelBaseDataLoader(config, labeled_set, 'train')
    unlabeled_loader = BaseParallelBaseDataLoader(config, unlabeled_set, 'train')
    dev_loader = BaseParallelBaseDataLoader(config, dev_set, 'test')

    print('labeled size: %d images, %d batches\n' % (len(labeled_indices), len(labeled_loader)),
          'unlabeled size: %d images, %d batches\n' % (len(unlabeled_indices), len(unlabeled_loader)),
          'dev size: %d images, %d batches\n' % (len(test_label), len(dev_loader)))

    return labeled_loader, unlabeled_loader, dev_loader
