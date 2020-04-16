import os
import numpy as np
from torchvision import datasets

from tensornet.data.datasets.dataset import BaseDataset


class MNIST(BaseDataset):
    """Load MNIST Dataset."""
    
    def _download(self, train=True, apply_transform=True):
        transform = None
        if apply_transform:
            transform = self.train_transform if train else self.val_transform
        return datasets.MNIST(
            self.path, train=train, download=True, transform=transform
        )
    
    def _get_classes(self):
        """Return list of classes in the dataset."""
        return tuple([
            str(x) for x in range(10)
        ])
    
    def _get_image_size(self):
        """Return shape of data i.e. image size."""
        return self.sample_data.data[0].unsqueeze(0).numpy().shape
    
    def _get_mean(self):
        return np.mean(self.sample_data.data.numpy()) / 255
    
    def _get_std(self):
        return np.std(self.sample_data.data.numpy()) / 255
