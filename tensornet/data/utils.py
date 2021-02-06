import torch
import numpy as np


def unnormalize(image, mean, std, transpose=False):
    """Un-normalize a given image.

    Args:
        image (:obj:`numpy.ndarray` or :obj:`torch.Tensor`): A ndarray
            or tensor. If tensor, it should be in CPU.
        mean (:obj:`float` or :obj:`tuple`): Mean. It can be a single value or
            a tuple with 3 values (one for each channel).
        std (:obj:`float` or :obj:`tuple`): Standard deviation. It can be a single
            value or a tuple with 3 values (one for each channel).
        transpose (:obj:`bool`, optional): If True, transposed output will
            be returned. This param is effective only when image is
            a tensor. If tensor, the output will have channel number
            as the last dim. (default: False)

    Returns:
        (`numpy.ndarray` or `torch.Tensor`): Unnormalized image
    """

    # Check if image is tensor, convert to numpy array
    tensor = False
    if type(image) == torch.Tensor:  # tensor
        tensor = True
        if len(image.size()) == 3:
            image = image.transpose(0, 1).transpose(1, 2)
        image = np.array(image)

    # Perform normalization
    image = image * std + mean

    # Convert image back to its original data type
    if tensor:
        if not transpose and len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        image = torch.Tensor(image)

    return image


def normalize(image, mean, std, transpose=False):
    """Normalize a given image.

    Args:
        image (:obj:`numpy.ndarray` or :obj:`torch.Tensor`): A ndarray
            or tensor. If tensor, it should be in CPU.
        mean (:obj:`float` or :obj:`tuple`): Mean. It can be a single value or
            a tuple with 3 values (one for each channel).
        std (:obj:`float` or :obj:`tuple`): Standard deviation. It can be a single
            value or a tuple with 3 values (one for each channel).
        transpose (:obj:`bool`, optional): If True, transposed output will
            be returned. This param is effective only when image is
            a tensor. If tensor, the output will have channel number
            as the last dim. (default: False)

    Returns:
        (`numpy.ndarray` or `torch.Tensor`): Normalized image
    """

    # Check if image is tensor, convert to numpy array
    tensor = False
    if type(image) == torch.Tensor:  # tensor
        tensor = True
        if len(image.size()) == 3:
            image = image.transpose(0, 1).transpose(1, 2)
        image = np.array(image)

    # Perform normalization
    image = (image - mean) / std

    # Convert image back to its original data type
    if tensor:
        if not transpose and len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        image = torch.Tensor(image)

    return image


def to_numpy(tensor):
    """Convert 3-D torch tensor to a 3-D numpy array.

    Args:
        tensor (torch.Tensor): Tensor to be converted.

    Returns:
        (*numpy.ndarray*): Image in numpy form.
    """
    return tensor.transpose(0, 1).transpose(1, 2).clone().numpy()


def to_tensor(ndarray):
    """Convert 3-D numpy array to 3-D torch tensor.

    Args:
        ndarray (numpy.ndarray): Array to be converted.

    Returns:
        (*torch.Tensor*): Image in tensor form.
    """
    return torch.Tensor(np.transpose(ndarray, (2, 0, 1)))
