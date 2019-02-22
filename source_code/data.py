import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import SVHN, CIFAR10


class DataLoader(object):

    def __init__(self, raw_loader, indices, batch_size):
        self.images, self.labels = [], []
        for idx in indices:
            image, label = raw_loader[idx]
            self.images.append(image)
            self.labels.append(label)

        self.images = torch.stack(self.images, 0)
        self.labels = torch.from_numpy(np.array(self.labels, dtype=np.int64)).squeeze()


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


def get_svhn_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = SVHN(config.data_root, split='train', download=True, transform=transform)
    dev_set = SVHN(config.data_root, split='test', download=True, transform=transform)

    def preprocess(data_set):
        for i in range(len(data_set.data)):
            if data_set.labels[i] == 10:
                data_set.labels[i] = 0

    preprocess(training_set)
    preprocess(dev_set)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(config.size_labeled_data / 10)]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size',
          len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    return labeled_loader, unlabeled_loader, dev_loader


def get_cifar_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = CIFAR10('cifar', train=True, download=True, transform=transform)
    dev_set = CIFAR10('cifar', train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(config.size_labeled_data / 10)]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size',
          len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    return labeled_loader, unlabeled_loader, dev_loader
