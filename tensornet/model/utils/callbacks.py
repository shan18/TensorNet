from torch.optim.lr_scheduler import StepLR


def lr_scheduler(optimizer, step_size, gamma):
    """Create LR scheduler.

    Args:
        optimizer: Model optimizer.
        step_size: Frequency for changing learning rate.
        gamma: Factor for changing learning rate.
    
    Returns:
        StepLR: Learning rate scheduler.
    """

    return StepLR(optimizer, step_size=step_size, gamma=gamma)
