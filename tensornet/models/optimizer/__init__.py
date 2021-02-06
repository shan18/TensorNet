import torch
import torch.optim as optim


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
