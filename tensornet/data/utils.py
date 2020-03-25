import torch
import numpy as np

def unnormalize(image, mean, std, transpose=False):
    """Un-normalize a given image.
    
    Args:
        image (numpy.ndarray or torch.Tensor): A ndarray
            or tensor. If tensor, it should be in CPU.
        mean (float or tuple): Mean. It can be a single value or
            a tuple with 3 values (one for each channel).
        std (float or tuple): Standard deviation. It can be a single
            value or a tuple with 3 values (one for each channel).
        transpose (bool, optional): If True, transposed output will
            be returned. This param is effective only when image is
            a tensor. If tensor, the output will have channel number
            as the last dim. (default: False)
    """
    if type(image) == torch.Tensor:  # tensor
        if transpose and len(image.size()) == 3:
            image = image.transpose(0, 1).transpose(1, 2)
        image = np.array(image)
    image = image * std + mean
    return image


def normalize(image, mean, std, transpose=False):
    """Normalize a given image.
    
    Args:
        image (numpy.ndarray or torch.Tensor): A ndarray
            or tensor. If tensor, it should be in CPU.
        mean (float or tuple): Mean. It can be a single value or
            a tuple with 3 values (one for each channel).
        std (float or tuple): Standard deviation. It can be a single
            value or a tuple with 3 values (one for each channel).
        transpose (bool, optional): If True, transposed output will
            be returned. This param is effective only when image is
            a tensor. If tensor, the output will have channel number
            as the last dim. (default: False)
    """
    if type(image) == torch.Tensor:  # tensor
        if transpose and len(image.size()) == 3:
            image = image.transpose(0, 1).transpose(1, 2)
        image = np.array(image)
    image = (image - mean) / std
    return image


def to_numpy(tensor):
    """Convert 3-D torch tensor to a 3-D numpy array.

    Args:
        tensor (torch.Tensor): Tensor to be converted.
    """
    return tensor.transpose(0, 1).transpose(1, 2).clone().numpy()


def to_tensor(ndarray):
    """Convert 3-D numpy array to 3-D torch tensor.

    Args:
        ndarray (numpy.ndarray): Array to be converted.
    """
    return torch.Tensor(np.transpose(ndarray, (2, 0, 1)))
