import torch
import numpy as np

from tensornet.data.downloader import download_mnist, download_cifar10
from tensornet.data.processing import Transformations, data_loader
from tensornet.data.utils import unnormalize, normalize


class Dataset:
    """Loads a dataset."""

    def __init__(
        self, train_batch_size=1, val_batch_size=1, cuda=False,
        num_workers=1, path=None, padding=(0, 0), crop=(0, 0),
        horizontal_flip_prob=0.0, vertical_flip_prob=0.0,
        gaussian_blur_prob=0.0, rotate_degree=0.0, cutout_prob=0.0,
        cutout_dim=(8, 8)
    ):
        """Initializes the dataset for loading.

        Args:
            train_batch_size (int, optional): Number of images to consider
                in each batch in train set. (default: 0)
            val_batch_size (int, optional): Number of images to consider
                in each batch in validation set. (default: 0)
            cuda (bool, optional): True is GPU is available. (default: False)
            num_workers (int, optional): How many subprocesses to use for
                data loading. (default: 0)
            path (str, optional): Path where dataset will be downloaded. If
                no path provided, data will be downloaded in a pre-defined
                directory. (default: None)
            padding (tuple, optional): Pad the image if the image size is less
                than the specified dimensions (height, width). (default: (0, 0))
            crop (tuple, optional): Randomly crop the image with the specified
                dimensions (height, width). (default: (0, 0))
            horizontal_flip_prob (float, optional): Probability of an image
                being horizontally flipped. (default: 0)
            vertical_flip_prob (float, optional): Probability of an image
                being vertically flipped. (default: 0)
            rotate_prob (float, optional): Probability of an image being rotated.
                (default: 0)
            rotate_degree (float, optional): Angle of rotation for image
                augmentation. (default: 0)
            cutout_prob (float, optional): Probability that cutout will be
                performed. (default: 0)
            cutout_dim (tuple, optional): Dimensions of the cutout box
                (height, width). (default: (8, 8))
        """
        
        self.cuda = cuda
        self.num_workers = num_workers
        self.path = path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        # Define classes present in the dataset
        self.class_values = None

        # Set data augmentation parameters
        self.padding = padding
        self.crop = crop
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.gaussian_blur_prob = gaussian_blur_prob
        self.rotate_degree = rotate_degree
        self.cutout_prob = cutout_prob
        self.cutout_dim = cutout_dim

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
            train (bool, optional): If True, download training data
                else download the test data. (default: True)
        
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
            args['padding'] = self.padding
            args['crop'] = self.crop
            args['horizontal_flip_prob'] = self.horizontal_flip_prob
            args['vertical_flip_prob'] = self.vertical_flip_prob
            args['gaussian_blur_prob'] = self.gaussian_blur_prob
            args['rotate_degree'] = self.rotate_degree
            args['cutout_prob'] = self.cutout_prob
            args['cutout_dim'] = self.cutout_dim

        return Transformations(**args)
    
    def _download(self, train=True, apply_transform=True):
        """Download dataset.

        Args:
            train (bool, optional): True for training data.
                (default: True)
            apply_transform (bool, optional): True if transform
                is to be applied on the dataset. (default: True)
        
        Returns:
            Downloaded dataset.
        """
        raise NotImplementedError
    
    def data(self, train=True):
        """Return data based on train mode.

        Args:
            train (bool, optional): True for training data. (default: True)
        
        Returns:
            Training or validation data and targets.
        """
        data = self.train_data if train else self.val_data
        return data.data, data.targets
    
    def unnormalize(self, image, transpose=False):
        """Un-normalize a given image.

        Args:
            image (numpy.ndarray or torch.Tensor): A ndarray
                or tensor. If tensor, it should be in CPU.
            transpose (bool, optional): If True, transposed output will
                be returned. This param is effective only when image is
                a tensor. If tensor, the output will have channel number
                as the last dim. (default: False)
        """
        return unnormalize(image, self.mean, self.std, transpose)
    
    def normalize(self, image, transpose=False):
        """Normalize a given image.

        Args:
            image (numpy.ndarray or torch.Tensor): A ndarray
                or tensor. If tensor, it should be in CPU.
            transpose (bool, optional): If True, transposed output will
                be returned. This param is effective only when image is
                a tensor. If tensor, the output will have channel number
                as the last dim. (default: False)
        """
        return normalize(image, self.mean, self.std, transpose)
    
    def loader(self, train=True):
        """Create data loader.

        Args:
            train (bool, optional): True for training data. (default: True)
        
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


class MNIST(Dataset):
    """Load MNIST Dataset."""
    
    def _download(self, train=True, apply_transform=True):
        transform = None
        if apply_transform:
            transform = self.train_transform if train else self.val_transform
        return download_mnist(self.path, train=train, transform=transform)
    
    @property
    def image_size(self):
        """Return shape of data i.e. image size."""
        return self.sample_data.data[0].unsqueeze(0).numpy().shape
    
    @property
    def mean(self):
        return np.mean(self.sample_data.data.numpy()) / 255
    
    @property
    def std(self):
        return np.std(self.sample_data.data.numpy()) / 255


class CIFAR10(Dataset):
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
        return download_cifar10(self.path, train=train, transform=transform)
    
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
