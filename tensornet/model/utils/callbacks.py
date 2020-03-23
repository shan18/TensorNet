from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def step_lr(optimizer, step_size, gamma):
    """Create LR step scheduler.

    Args:
        optimizer: Model optimizer.
        step_size: Frequency for changing learning rate.
        gamma: Factor for changing learning rate.
    
    Returns:
        StepLR: Learning rate scheduler.
    """

    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def reduce_lr_on_plateau(optimizer, factor=0.1, patience=10, verbose=False, min_lr=0):
    """Create LR plateau reduction scheduler.

    Args:
        optimizer: Model optimizer.
        factor: Factor by which the learning rate will be reduced.
        patience: Number of epoch with no improvement after which learning
            rate will be will be reduced.
        verbose: If True, prints a message to stdout for each update.
            Defaults to False.
        min_lr: A scalar or a list of scalars. A lower bound on the
            learning rate of all param groups or each group respectively.
            Defaults to 0.
    
    Returns:
        ReduceLROnPlateau instance.
    """

    return ReduceLROnPlateau(
        optimizer, factor=factor, patience=patience, verbose=verbose, min_lr=min_lr
    )
