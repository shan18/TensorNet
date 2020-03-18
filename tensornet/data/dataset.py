import torch
import numpy as np

from tensornet.data.downloader import download_cifar10
from tensornet.data.processing import Transformations, data_loader
from tensornet.data.utils import unnormalize


class CIFAR10:
    """ Load CIFAR-10 Dataset. """

    def __init__(
        self, train_batch_size=1, val_batch_size=1, cuda=False,
        num_workers=1, path=None, horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0, gaussian_blur_prob=0.0,
        rotate_degree=0.0, cutout=0.0
    ):
        """Initializes the dataset for loading.

        Args:
            train_batch_size: Number of images to consider in each batch in train set.
            val_batch_size: Number of images to consider in each batch in validation set.
            cuda: True is GPU is available.
            num_workers: How many subprocesses to use for data loading.
            path: Path where dataset will be downloaded. Defaults to None.
                If no path provided, data will be downloaded in a pre-defined
                directory.
            horizontal_flip_prob: Probability of an image being horizontally flipped.
                Defaults to 0.
            vertical_flip_prob: Probability of an image being vertically flipped.
                Defaults to 0.
            rotate_prob: Probability of an image being rotated.
                Defaults to 0.
            rotate_degree: Angle of rotation for image augmentation.
                Defaults to 0.
            cutout: Probability that cutout will be performed.
                Defaults to 0.
        """
        
        self.cuda = cuda
        self.num_workers = num_workers
        self.path = path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        # Define classes present in the dataset
        self.class_values = (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        )

        # Set data augmentation parameters
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.gaussian_blur_prob = gaussian_blur_prob
        self.rotate_degree = rotate_degree
        self.cutout = cutout

        # Download sample data
        # This is done to get the image size
        # and mean and std of the dataset
        self.sample_data = self._download(apply_transform=False)

        # Set training data
        self.train_transform = self._transform()
        self.train_data = self._download()

        # Set validation data
        self.val_transform = self._transform(train=False)
        self.val_data = self._download(train=False)
    
    def _transform(self, train=True):
        """Define data transformations
        
        Args:
            train: If True, download training data else test data.
                Defaults to True.
        
        Returns:
            Returns data transforms based on the training mode.
        """

        args = {
            'mean': self.mean,
            'std': self.std,
            'train': False
        }

        if train:
            args['train'] = True
            args['horizontal_flip_prob'] = self.horizontal_flip_prob
            args['vertical_flip_prob'] = self.vertical_flip_prob
            args['gaussian_blur_prob'] = self.gaussian_blur_prob
            args['rotate_degree'] = self.rotate_degree
            args['cutout'] = self.cutout
            args['cutout_height'] = self.image_size[1] // 2
            args['cutout_width'] = self.image_size[2] // 2

        return Transformations(**args)
    
    def _download(self, train=True, apply_transform=True):
        """Download dataset.

        Args:
            train: True for training data.
            apply_transform: True if transform is to be applied on the data.
        
        Returns:
            Downloaded dataset.
        """
        transform = None
        if apply_transform:
            transform = self.train_transform if train else self.val_transform
        return download_cifar10(self.path, train=train, transform=transform)
    
    @property
    def classes(self):
        """ Return list of classes in the dataset. """
        return self.class_values
    
    @property
    def image_size(self):
        """ Return shape of data i.e. image size. """
        return np.transpose(self.sample_data.data[0], (2, 0, 1)).shape
    
    @property
    def mean(self):
        return tuple(np.mean(self.sample_data.data, axis=(0, 1, 2)) / 255)
    
    @property
    def std(self):
        return tuple(np.std(self.sample_data.data, axis=(0, 1, 2)) / 255)
    
    def data(self, train=True):
        """ Return data based on train mode.

        Args:
            train: True for training data.
        
        Returns:
            Training or validation data and targets.
        """
        data = self.train_data if train else self.val_data
        return data.data, data.targets
    
    def unnormalize(self, image, out_type='array'):
        """Un-normalize a given image.

        Args:
            image: A 3-D ndarray or 3-D tensor.
                If tensor, it should be in CPU.
        """
        return unnormalize(image, self.mean, self.std, out_type)
    
    def loader(self, train=True):
        """Create data loader.

        Args:
            train: True for training data.
        
        Returns:
            Dataloader instance.
        """

        loader_args = {
            'batch_size': self.train_batch_size if train else self.val_batch_size,
            'num_workers': self.num_workers,
            'cuda': self.cuda
        }

        return data_loader(
            self.train_data, **loader_args
        ) if train else data_loader(self.val_data, **loader_args)
