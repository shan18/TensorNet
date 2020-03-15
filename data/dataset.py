import numpy as np

from data.downloader import download_cifar10
from data.processing import transformations, data_loader


class CIFAR10:
    """ Load CIFAR-10 Dataset. """

    def __init__(
        self, train_batch_size=1, val_batch_size=1,
        cuda=False, num_workers=1, path=None,
        horizontal_flip=0.0, vertical_flip=0.0, rotation=0.0, random_erasing=0.0
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
            horizontal_flip: Probability of an image being horizontally flipped.
                Defaults to 0.
            vertical_flip: Probability of an image being vertically flipped.
                Defaults to 0.
            rotation: Angle of rotation for image augmentation.
                Defaults to 0.
            random_erasing: Probability that random erase will be performed.
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
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.random_erasing = random_erasing

        # Set transforms
        self.train_transform = self._transform()
        self.val_transform = self._transform(train=False)

        # Download dataset
        self.train_data = self._download()
        self.val_data = self._download(train=False)
    
    def _transform(self, train=True):
        """Define data transformations
        
        Args:
            train: If True, download training data else test data.
                Defaults to True.
        
        Returns:
            Returns data transforms based on the training mode.
        """
        return transformations(
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            rotation=self.rotation,
            random_erasing=self.random_erasing
        ) if train else transformations()
    
    def _download(self, train=True):
        """Download dataset.

        Args:
            train: True for training data.
        
        Returns:
            Downloaded dataset.
        """
        transform = self.train_transform if train else self.val_transform
        return download_cifar10(self.path, train=train, transform=transform)
    
    def classes(self):
        """ Return list of classes in the dataset. """
        return self.class_values
    
    def data(self, train=True):
        """ Return data based on train mode.

        Args:
            train: True for training data.
        
        Returns:
            Training or validation data and targets.
        """
        data = self.train_data if train else self.val_data
        return data.data, data.targets
    
    def image_size(self):
        """ Return shape of data i.e. image size. """
        return np.transpose(self.data()[0][0], (2, 0, 1)).shape
    
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
