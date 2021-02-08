import torch
import torch.optim as optim
from typing import Tuple


def sgd(
    model: torch.nn.Module,
    learning_rate: float = 0.01,
    momentum: int = 0,
    dampening: int = 0,
    l2_factor: float = 0.0,
    nesterov: bool = False,
):
    """SGD optimizer.

    Args:
        model (torch.nn.Module): Model Instance.
        learning_rate (:obj:`float`, optional): Learning rate for the optimizer. (default: 0.01)
        momentum (:obj:`float`, optional): Momentum factor. (default: 0)
        dampening (:obj:`float`, optional): Dampening for momentum. (default: 0)
        l2_factor (:obj:`float`, optional): Factor for L2 regularization. (default: 0)
        nesterov (:obj:`bool`, optional): Enables nesterov momentum. (default: False)

    Returns:
        SGD optimizer.
    """
    return optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        dampening=dampening,
        weight_decay=l2_factor,
        nesterov=nesterov
    )


def adam(
    model: torch.nn.Module,
    learning_rate: float = 0.001,
    betas: Tuple[float] = (0.9, 0.999),
    eps: float = 1e-08,
    l2_factor: float = 0.0,
    amsgrad: bool = False,
):
    """Adam optimizer.

    Args:
        model (torch.nn.Module): Model Instance.
        learning_rate (:obj:`float`, optional): Learning rate for the optimizer. (default: 0.001)
        betas (:obj:`tuple`, optional): Coefficients used for computing running averages of
            gradient and its square. (default: (0.9, 0.999))
        eps (:obj:`float`, optional): Term added to the denominator to improve numerical stability.
            (default: 1e-8)
        l2_factor (:obj:`float`, optional): Factor for L2 regularization. (default: 0)
        amsgrad (:obj:`bool`, optional): Whether to use the AMSGrad variant of this algorithm from the
            paper `On the Convergence of Adam and Beyond <https://openreview.net/forum?id=ryQu7f-RZ>`_.
            (default: False)

    Returns:
        Adam optimizer.
    """
    return optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=l2_factor,
        amsgrad=amsgrad
    )
