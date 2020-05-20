import torch.nn as nn

from .ssim import SSIMLoss, MSSSIMLoss


def cross_entropy_loss():
    """Create Cross Entropy Loss.
    The loss automatically applies the softmax activation
    function on the prediction input.

    Returns:
        Cross entroy loss function
    """
    return nn.CrossEntropyLoss()


def bce_loss():
    """Create Binary Cross Entropy Loss.
    The loss automatically applies the sigmoid activation
    function on the prediction input.

    Returns:
        Binary cross entropy loss function
    """
    return nn.BCEWithLogitsLoss()


def mse_loss():
    """Create Mean Squared Error Loss.

    Returns:
        Mean squared error loss function
    """
    return nn.MSELoss()


def ssim_loss(data_range=1.0, size_average=True, channel=3):
    """Create SSIM Loss.

    Args:
        data_range (float or int, optional): Value range of input
            images (usually 1.0 or 255). (default: 255)
        size_average (bool, optional): If size_average=True, ssim
            of all images will be averaged as a scalar. (default: True)
        channel (int, optional): input channels (default: 3)

    Returns:
        SSIM loss function
    """
    return SSIMLoss(
        data_range=data_range, size_average=size_average, channel=channel
    )


def ms_ssim_loss(data_range=1.0, size_average=True, channel=3):
    """Create MS-SSIM Loss.

    Args:
        data_range (float or int, optional): Value range of input
            images (usually 1.0 or 255). (default: 1.0)
        size_average (bool, optional): If size_average=True, ssim
            of all images will be averaged as a scalar. (default: True)
        channel (int, optional): input channels (default: 3)

    Returns:
        MS-SSIM loss function
    """
    return MSSSIMLoss(
        data_range=data_range, size_average=size_average, channel=channel
    )
