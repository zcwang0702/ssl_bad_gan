import numpy as np
from torchvision import transforms
from torchvision import datasets
from .base_data_loader import BaseDataLoader


def svhn_label_preprocess(data_set):
    for i in range(len(data_set.data)):
        if data_set.labels[i] == 10:
            data_set.labels[i] = 0


def get_ssl_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    name_config = config['data_loader']['type']
    arg_config = config['data_loader']['args']
    train_config = {
        'root': '../data/%s'%(name_config),
        'split': 'train',
        'transform': transform,
        "download": True,
    }
    dev_config = {
        'root': '../data/%s'%(name_config),
        'split': 'test',
        'transform': transform,
        "download": True,
    }

    training_set = getattr(datasets, name_config)(**train_config)
    dev_set = getattr(datasets, name_config)(**dev_config)

    if config['data_loader']['type'] == 'SVHN':
        svhn_label_preprocess(training_set)
        svhn_label_preprocess(dev_set)

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

    a, b, c = get_ssl_loaders(json.load(open('../config/config.json')))
    print(len(a))
