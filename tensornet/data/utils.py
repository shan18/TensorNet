import torch
import numpy as np


def unnormalize(image, mean, std, transpose=False):
    """Un-normalize a given image.
    
    Args:
        image: A 3-D ndarray or 3-D tensor.
            If tensor, it should be in CPU.
        mean: Mean value. It can be a single value or
            a tuple with 3 values (one for each channel).
        std: Standard deviation value. It can be a single value or
            a tuple with 3 values (one for each channel).
        transpose: If True, transposed output will be returned.
            This param is effective only when image is a tensor.
            If tensor, the output will have channel number
            as the last dim.
    """

    if not type(mean) in [list, tuple]:
        mean = (mean,) * 3
    if not type(std) in [list, tuple]:
        std = (std,) * 3

    if type(image) == torch.Tensor:  # tensor
        image = image * torch.Tensor(std)[:, None, None] + torch.Tensor(mean)[:, None, None]
        if transpose:
            image = image.transpose(0, 1).transpose(1, 2)
    elif type(image) == np.ndarray:  # numpy array
        image = image * std + mean
    else:  # No valid value given
        image = None
    
    return image


def normalize(image, mean, std, transpose=False):
    """Normalize a given image.
    
    Args:
        image: A 3-D ndarray or 3-D tensor.
            If tensor, it should be in CPU.
        mean: Mean value. It can be a single value or
            a tuple with 3 values (one for each channel).
        std: Standard deviation value. It can be a single value or
            a tuple with 3 values (one for each channel).
        transpose: If True, transposed output will be returned.
            This param is effective only when image is a tensor.
            If tensor, the output will have channel number
            as the last dim.
    """

    if not type(mean) in [list, tuple]:
        mean = (mean,) * 3
    if not type(std) in [list, tuple]:
        std = (std,) * 3

    if type(image) == torch.Tensor:  # tensor
        image = (image - torch.Tensor(mean)[:, None, None]) / torch.Tensor(std)[:, None, None]
        if transpose:
            image = image.transpose(0, 1).transpose(1, 2)
    elif type(image) == np.ndarray:  # numpy array
        image = (image - mean) / std
    else:  # No valid value given
        image = None
    
    return image


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
