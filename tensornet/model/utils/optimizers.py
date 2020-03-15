import torch.optim as optim


def sgd_optimizer(model, learning_rate, momentum, l2_factor=0.0):
    """Create optimizer.

    Args:
        model: Model instance.
        learning_rate: Learning rate for the optimizer.
        momentum: Momentum of optimizer.
        l2_factor: Factor for L2 regularization.
    
    Returns:
        SGD optimizer.
    """
    return optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2_factor
    )
