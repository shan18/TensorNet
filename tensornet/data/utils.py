import torch
import numpy as np


def unnormalize(image, mean, std):
    """Un-normalize a given image.
    
    Args:
        image: A numpy array with shape (H, W, 3).
        mean: Mean value. It can be a single value or
            a tuple with 3 values (one for each channel).
        std: Standard deviation value. It can be a single value or
            a tuple with 3 values (one for each channel).
    """

    return image * std + mean


def to_numpy(tensor):
    """Convert 3-D torch tensor to a 3-D numpy array.

    Args:
        tensor: Tensor to be converted.
    """
    return np.transpose(tensor.clone().numpy(), (1, 2, 0))


def to_tensor(ndarray):
    """Convert 3-D numpy array to 3-D torch tensor.

    Args:
        ndarray: Array to be converted.
    """
    return torch.Tensor(np.transpose(ndarray, (2, 0, 1)))
