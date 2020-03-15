import torch.nn as nn


def cross_entropy_loss():
    """Create Cross Entropy Loss

    Returns:
        Cross entroy loss function
    """
    return nn.CrossEntropyLoss()
