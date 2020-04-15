import os
import numpy as np
from torchvision import datasets

from tensornet.data.datasets.dataset import BaseDataset


class CIFAR10(BaseDataset):
    """Load CIFAR-10 Dataset."""
    
    @property
    def classes(self):
        """Return list of classes in the dataset."""
        return (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        )
    
    def _download(self, train=True, apply_transform=True):
        transform = None
        if apply_transform:
            transform = self.train_transform if train else self.val_transform
        return datasets.CIFAR10(
            self.path, train=train, download=True, transform=transform
        )
    
    @property
    def image_size(self):
        """Return shape of data i.e. image size."""
        return np.transpose(self.sample_data.data[0], (2, 0, 1)).shape
    
    @property
    def mean(self):
        return tuple(np.mean(self.sample_data.data, axis=(0, 1, 2)) / 255)
    
    @property
    def std(self):
        return tuple(np.std(self.sample_data.data, axis=(0, 1, 2)) / 255)
