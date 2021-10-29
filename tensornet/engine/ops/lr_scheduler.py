from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR, CyclicLR


def step_lr(optimizer, step_size, gamma=0.1, last_epoch=-1):
    """Create LR step scheduler.

    Args:
        optimizer (torch.optim): Model optimizer.
        step_size (int): Frequency for changing learning rate.
        gamma (:obj:`float`, optional): Factor for changing learning rate. (default: 0.1)
        last_epoch (:obj:`int`, optional): The index of last epoch. (default: -1)

    Returns:
        StepLR: Learning rate scheduler.
    """

    return StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)


def reduce_lr_on_plateau(optimizer, factor=0.1, patience=10, verbose=False, min_lr=0):
    """Create LR plateau reduction scheduler.

    Args:
        optimizer (torch.optim): Model optimizer.
        factor (:obj:`float`, optional): Factor by which the learning rate will be reduced.
            (default: 0.1)
        patience (:obj:`int`, optional): Number of epoch with no improvement after which learning
            rate will be will be reduced. (default: 10)
        verbose (:obj:`bool`, optional): If True, prints a message to stdout for each update.
            (default: False)
        min_lr (:obj:`float`, optional): A scalar or a list of scalars. A lower bound on the
            learning rate of all param groups or each group respectively. (default: 0)

    Returns:
        ReduceLROnPlateau instance.
    """

    return ReduceLROnPlateau(
        optimizer, factor=factor, patience=patience, verbose=verbose, min_lr=min_lr
    )


def one_cycle_lr(
    optimizer, max_lr, epochs, steps_per_epoch, pct_start=0.5, div_factor=10.0, final_div_factor=10000
):
    """Create One Cycle Policy for Learning Rate.

    Args:
        optimizer (torch.optim): Model optimizer.
        max_lr (float): Upper learning rate boundary in the cycle.
        epochs (int): The number of epochs to train for. This is used along with
            steps_per_epoch in order to infer the total number of steps in the cycle.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        pct_start (:obj:`float`, optional): The percentage of the cycle (in number of steps)
            spent increasing the learning rate. (default: 0.5)
        div_factor (:obj:`float`, optional): Determines the initial learning rate via
            initial_lr = max_lr / div_factor. (default: 10.0)
        final_div_factor (:obj:`float`, optional): Determines the minimum learning rate via
            min_lr = initial_lr / final_div_factor. (default: 1e4)

    Returns:
        OneCycleLR instance.
    """

    return OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor
    )


def cyclic_lr(
    optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0,
    scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
    last_epoch=- 1, verbose=False
):
    """Create Cyclic LR Policy.

    Args:
        optimizer (torch.optim): Model optimizer.
        base_lr (float): Lower learning rate boundary in the cycle.
        max_lr (float): Upper learning rate boundary in the cycle.
        step_size_up (int): Number of training iterations in the increasing half of a cycle.
            (default: 2000)
        step_size_down (int): Number of training iterations in the decreasing half of a cycle.
            If step_size_down is None, it is set to step_size_up. (default: None)
        mode (str): One of `triangular`, `triangular2`, `exp_range`. If scale_fn is not None,
            this argument is ignored. (default: ‘triangular’)
        gamma (float): Constant in ‘exp_range’ scaling function: gamma**(cycle iterations).
            (default: 1.0)
        scale_fn: Custom scaling policy defined by a single argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0. If specified, then ‘mode’ is ignored.
            (default: None)
        scale_mode (str): ‘cycle’, ‘iterations’. Defines whether scale_fn is evaluated on cycle
            number or cycle iterations (training iterations since start of cycle).
            (default: ‘cycle’)
        cycle_momentum (bool): If True, momentum is cycled inversely to learning rate between
            ‘base_momentum’ and ‘max_momentum’. (default: True)
        base_momentum (float): Lower momentum boundaries in the cycle. (default: 0.8)
        max_momentum (float): Upper momentum boundaries in the cycle. Functionally, it defines
            the cycle amplitude (max_momentum - base_momentum). (default: 0.9)
        last_epoch (int): The index of the last batch. This parameter is used when resuming a
            training job.(default: -1)
        verbose (bool): If True, prints a message to stdout for each update. (default: False)

    Returns:
        CyclicLR instance.
    """
    return CyclicLR(
        optimizer, base_lr, max_lr, step_size_up=step_size_up, step_size_down=step_size_down,
        mode=mode, gamma=gamma, scale_fn=scale_fn, scale_mode=scale_mode, cycle_momentum=cycle_momentum,
        base_momentum=base_momentum, max_momentum=max_momentum, last_epoch=last_epoch, verbose=verbose
    )
