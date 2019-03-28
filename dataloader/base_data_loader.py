import numpy as np
import torch
from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):

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
                if (end - start) != self.batch_size:  # drop the last batch
                    continue
                ret_images, ret_labels = self.images[indices[start: end]], self.labels[indices[start: end]]
                yield ret_images, ret_labels
            if not inf:
                break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return self.len


class BaseParallelBaseDataLoader(DataLoader):
    def __init__(self, config, dataset, phase):
        self.config = config
        self.dataloader_config = config['data_loader']['args']

        self.para_dict = {'shuffle': phase == 'train',
                          'pin_memory': self.config['n_gpu'] > 0,

                          'batch_size': self.dataloader_config['train_batch_size'] if phase == 'train' \
                              else self.dataloader_config['dev_batch_size'],

                          'num_workers': self.dataloader_config['num_workers'],
                          'drop_last': self.dataloader_config['drop_last']}

        super(BaseParallelBaseDataLoader, self).__init__(dataset, **self.para_dict)

