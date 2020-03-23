# The code in this file is referenced from https://github.com/davidtvs/pytorch-lr-finder


import os
import copy
import torch
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler


class LRFinder(object):
    """Learning rate range test.
    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.

    Args:
        model: Model Instance.
        optimizer: optimizer where the defined learning
            is assumed to be the lower boundary of the range test.
        criterion: wrapped loss function.
        device: a string ('cpu or 'cuda') with an optional ordinal for the device type
            (e.g. 'cuda:X', where is the ordinal). Alternatively, can be an object
            representing the device on which the computation will take place.
            Default: None, uses the same device as 'model'.
        memory_cache: if this flag is set to True, 'state_dict' of
            model and optimizer will be cached in memory. Otherwise, they will be saved
            to files under the 'cache_dir'.
        cache_dir: path for storing temporary files. If no path is
            specified, system-wide temporary directory is used. Notice that this
            parameter will be ignored if 'memory_cache' is True.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device=None,
        memory_cache=True,
        cache_dir=None,
    ):
        # Check if the optimizer is already attached to a scheduler
        self.optimizer = optimizer
        self._check_for_scheduler()

        self.model = model
        self.criterion = criterion
        self.history = {'lr': [], 'loss': []}
        self.best_loss = None
        self.best_lr = None
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir

        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_device = next(self.model.parameters()).device
        self.state_cacher = StateCacher(memory_cache, cache_dir=cache_dir)
        self.state_cacher.store('model', self.model.state_dict())
        self.state_cacher.store('optimizer', self.optimizer.state_dict())

        # If device is None, use the same as the model
        self.device = self.model_device if not device else device

    def reset(self):
        """Restores the model and optimizer to their initial states."""

        self.model.load_state_dict(self.state_cacher.retrieve('model'))
        self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer'))
        self.model.to(self.model_device)

    def _check_for_scheduler(self):
        """Check if the optimizer has and existing scheduler attached to it."""
        for param_group in self.optimizer.param_groups:
            if 'initial_lr' in param_group:
                raise RuntimeError('Optimizer already has a scheduler attached to it')

    def _set_learning_rate(self, new_lrs):
        """Set the given learning rates in the optimizer."""
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                'Length of new_lrs is not equal to the number of parameter groups in the given optimizer'
            )

        # Set the learning rates to the parameter groups
        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr

    def range_test(
        self,
        train_loader,
        val_loader=None,
        start_lr=None,
        end_lr=10,
        num_iter=None,
        step_mode='exp',
        smooth_f=0.05,
        diverge_th=5,
    ):
        """Performs the learning rate range test.

        Args:
            train_loader (torch.utils.data.DataLoader): the training set data laoder.
            val_loader (torch.utils.data.DataLoader, optional): if None the range test
                will only use the training loss. When given a data loader, the model is
                evaluated after each iteration on that dataset and the evaluation loss
                is used. Note that in this mode the test takes significantly longer but
                generally produces more precise results. Default: None.
            start_lr (float, optional): the starting learning rate for the range test.
                Default: None (uses the learning rate from the optimizer).
            end_lr (float, optional): the maximum learning rate to test. Default: 10.
            num_iter (int, optional): the number of iterations over which the test
                occurs. If None, then test occurs for one epoch. Default is None.
            step_mode (str, optional): one of the available learning rate policies,
                linear or exponential ("linear", "exp"). Default: "exp".
            smooth_f (float, optional): the loss smoothing factor within the [0, 1]
                interval. Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th (int, optional): the test is stopped when the loss surpasses the
                threshold:  diverge_th * best_loss. Default: 5.
        """

        # Reset test results
        self.history = {'lr': [], 'loss': []}
        self.best_loss = None
        self.best_lr = None

        # Move the model to the proper device
        self.model.to(self.device)

        # Check if the optimizer is already attached to a scheduler
        self._check_for_scheduler()

        # Set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)
        
        # Set number of iterations
        if num_iter is None:
            num_iter = len(train_loader.dataset) / train_loader.batch_size

        # Initialize the proper learning rate policy
        if step_mode.lower() == 'exp':
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == 'linear':
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError(f'Expected one of (exp, linear), got {step_mode}')

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError('smooth_f is outside the range [0, 1]')

        # Create an iterator to get data batch by batch
        train_iterator = iter(train_loader)
        pbar = tqdm(range(num_iter))
        for _, iteration in enumerate(pbar, 0):
            # Train on batch and retrieve loss
            loss = self._train_batch(train_iterator)
            if val_loader:
                loss = self._validate(val_loader)

            # Update the learning rate
            lr_schedule.step()
            self.history['lr'].append(lr_schedule.get_lr()[0])

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                self.best_loss = loss
                self.best_lr = self.history['lr'][-1]
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history['loss'][-1]
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_lr = self.history['lr'][-1]

            # Check if the loss has diverged; if it has, stop the test
            self.history['loss'].append(loss)
            if loss > diverge_th * self.best_loss:
                pbar.update(num_iter)
                print('Stopping early, the loss has diverged')
                break

        print('Learning rate search finished.')

    def _train_batch(self, train_iterator):
        self.model.train()
        total_loss = None  # for late initialization

        self.optimizer.zero_grad()
        inputs, labels = next(train_iterator)
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _validate(self, loader):
        # Set model to evaluation mode and disable gradient computation
        loss = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                # Move data to the correct device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.size(0)

                # Forward pass and loss computation
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item() * batch_size

        return loss / len(loader.dataset)

    def plot(self, skip_start=10, skip_end=5, log_lr=True, show_lr=None):
        """Plots the learning rate range test.

        Args:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the end.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
            show_lr (float, optional): is set, will add vertical line to visualize
                specified learning rate; Default: None.
        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
        if show_lr is not None and not isinstance(show_lr, float):
            raise ValueError("show_lr must be float")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history['lr']
        losses = self.history['loss']
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')

        if show_lr is not None:
            plt.axvline(x=show_lr, color='red')
        plt.show()


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    
    Args:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.

    Args:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class StateCacher(object):
    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile

            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError("Given cache_dir is not a valid directory.")

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, f'state_{key}_{id(self)}.pt')
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError(f'Target {key} was not cached.')

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError(
                    f"Failed to load state in {fn}. File doesn't exist anymore."
                )
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in cache_dir before
        this instance being destroyed.
        """

        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])
