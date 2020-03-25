import os

from torchvision import datasets


def download_cifar10(path=None, train=True, transform=None):
    """Download CIFAR10 dataset

    Args:
        path (str, optional): Path where dataset will be downloaded.
            If no path provided, data will be downloaded in a pre-defined
            directory. (default: None)
        train (bool, optional): If True, download the training data else
            download the test data. (default: True)
        transform (tensornet.Transformations, optional): Data transformations
            to be applied on the data. (default: None)
    
    Returns:
        Downloaded dataset.
    """

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cifar10')
    return datasets.CIFAR10(
        path, train=train, download=True, transform=transform
    )


def download_mnist(path=None, train=True, transform=None):
    """Download MNIST dataset

    Args:
        path (str, optional): Path where dataset will be downloaded.
            If no path provided, data will be downloaded in a pre-defined
            directory. (default: None)
        train (bool, optional): If True, download the training data else
            download the test data. (default: True)
        transform (tensornet.Transformations, optional): Data transformations
            to be applied on the data. (default: None)
    
    Returns:
        Downloaded dataset.
    """

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist')
    return datasets.MNIST(
        path, train=train, download=True, transform=transform
    )
