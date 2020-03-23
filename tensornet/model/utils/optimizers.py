import torch.optim as optim


def sgd(model, learning_rate=0.01, momentum=0, dampening=0, l2_factor=0.0, nesterov=False):
    """Create optimizer.

    Args:
        model: Model instance.
        learning_rate: Learning rate for the optimizer. Defaults to 0.01.
        momentum: Momentum factor. Defaults to 0.
        dampening: Dampening for momentum. Defaults to 0.
        l2_factor: Factor for L2 regularization. Defaults to 0.
        nesterov: Enables nesterov momentum. Defaults to False.
    
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
