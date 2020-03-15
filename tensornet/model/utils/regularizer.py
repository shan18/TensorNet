import torch
import torch.nn as nn


def l1(model, loss, factor):
    """Apply L1 regularization.

    Args:
        model: Model instance.
        loss: Loss function value.
        factor: Factor for applying L1 regularization
    
    Returns:
        Regularized loss value.
    """

    if factor > 0:
        criteria = nn.L1Loss(size_average=False)
        regularizer_loss = 0
        for parameter in model.parameters():
            regularizer_loss += criteria(parameter, torch.zeros_like(parameter))
        loss += factor * regularizer_loss
    return loss
