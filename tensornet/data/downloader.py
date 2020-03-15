import os

from torchvision import datasets


def download_cifar10(path, train=True, transform=None):
    """Download CIFAR10 dataset

    Args:
        path: Path where dataset will be downloaded. Defaults to None.
            If no path provided, data will be downloaded in a pre-defined
            directory.
        train: If True, download training data else test data.
            Defaults to True.
        transform: Data transformations to be applied on the data.
            Defaults to None.
    
    Returns:
        Downloaded dataset.
    """

    if not path:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cifar10')
    return datasets.CIFAR10(
        path, train=train, download=True, transform=transform
    )
