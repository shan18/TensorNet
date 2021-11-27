from torchvision import datasets

from tensornet.data.datasets.dataset import BaseDataset


class CIFAR100(BaseDataset):
    """CIFAR-100 Dataset.

    `Note`: This dataset inherits the ``BaseDataset`` class.
    """

    def _download(self, train=True, apply_transform=True):
        """Download dataset.

        Args:
            train (:obj:`bool`, optional): True for training data.
                (default: True)
            apply_transform (:obj:`bool`, optional): True if transform
                is to be applied on the dataset. (default: True)

        Returns:
            Downloaded dataset.
        """
        transform = None
        if apply_transform:
            transform = self.train_transform if train else self.val_transform
        return datasets.CIFAR100(
            self.path, train=train, download=True, transform=transform
        )

    def _get_image_size(self):
        """Return shape of data i.e. image size."""
        return (3, 32, 32)

    def _get_classes(self):
        """Return list of classes in the dataset."""
        return (
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
            'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
            'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
            'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
            'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
            'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        )

    def _get_mean(self):
        """Returns mean of the entire dataset."""
        return (0.5071, 0.4867, 0.4408)

    def _get_std(self):
        """Returns standard deviation of the entire dataset."""
        return (0.2675, 0.2565, 0.2761)
