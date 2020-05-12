import torch
import torch.nn.functional as F

from tensornet.engine.ops.regularizer import l1
from tensornet.engine.ops.checkpoint import ModelCheckpoint
from tensornet.data.processing import InfiniteDataLoader
from tensornet.utils.progress_bar import ProgressBar


class Learner:

    def __init__(
        self, model, optimizer, criterion, train_loader, device='cpu',
        epochs=1, val_loader=None, l1_factor=0.0, callbacks=None, metric=None
    ):
        """Train and validate the model.

        Args:
            model (torch.nn.Module): Model Instance.
            optimizer (torch.optim): Optimizer for the model.
            criterion (torch.nn): Loss Function.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            device (str or torch.device, optional): Device where the data
                will be loaded. (default='cpu')
            epochs (int, optional): Numbers of epochs/iterations to train the model for.
                (default: 1)
            val_loader (torch.utils.data.DataLoader, optional): Validation data
                loader. (default: None)
            l1_factor (float, optional): L1 regularization factor. (default: 0)
            callbacks (list, optional): List of callbacks to be used during training.
                (default: None)
            metric (str or tuple, optional): tuple or 'accuracy' for model evaluation. If
                tuple, then first element is the metric name and second element is the
                function for metric calculation. (default: None)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.val_loader = val_loader
        self.l1_factor = l1_factor

        self.metric = None
        self.metric_fn = None
        if metric:
            self._setup_metric(metric)
        
        self.lr_schedulers = {
            'step_lr': None,
            'lr_plateau': None,
            'one_cycle_policy': None,
        }
        self.checkpoint = None
        if not callbacks is None:
            self._setup_callbacks(callbacks)

        # Training
        self.train_losses = []  # Change in loss
        self.train_metric = []  # Change in accuracy

        self.val_losses = []  # Change in loss
        self.val_metric = []  # Change in accuracy
    
    def _setup_callbacks(self, callbacks):
        """Extract callbacks passed to the class."""
        for callback in callbacks:
            if isinstance(callback, torch.optim.lr_scheduler.StepLR):
                self.lr_schedulers['step_lr'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_schedulers['lr_plateau'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.OneCycleLR):
                self.lr_schedulers['one_cycle_policy'] = callback
            elif isinstance(callback, ModelCheckpoint):
                self.checkpoint = callback
    
    def _setup_metric(self, metric):
        """Validate the evaluation metric passed to the class."""
        self.metric_val = None
        if isinstance(metric, str):
            if metric != 'accuracy':
                raise ValueError(f'Invalid metric {metric} specified.')
            else:
                self.correct = 0  # Total number of correctly predicted samples so far
                self.processed = 0  # Total number of predicted samples so far
                self.metric = metric
                self.metric_fn = self._accuracy
        elif isinstance(metric, (list, tuple)):
            self.metric = metric[0]
            self.metric_fn = metric[1]
        else:
            raise ValueError('Invalid metric given.')
    
    def _reset_metric(self):
        """Reset metric params."""
        self.metric_val = None
        if self.metric == 'accuracy':
            self.correct = 0
            self.processed = 0
    
    def _accuracy(self, label, prediction):
        """Calculate accuracy.
        
        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        
        Returns:
            accuracy
        """
        pred_max = prediction.argmax(dim=1, keepdim=True)
        self.correct += pred_max.eq(
            label.view_as(pred_max)
        ).sum().item()
        self.processed += len(label)
        self.metric_val = round(
            100 * self.correct / self.processed, 2
        )
    
    def _get_pbar_values(self, **kwargs):
        return [
            (x, y) for x, y in kwargs.items()
        ]

    def update_training_history(self, loss):
        """Update the training history."""
        self.train_losses.append(loss)
        if self.metric_fn:
            self.train_metric.append(self.metric_val)
    
    def reset_history(self):
        """Reset the training history"""
        self.train_losses = []
        self.train_metric = []
        self.val_losses = []
        self.val_metric = []
        self._reset_metric()
    
    def train_batch(self, data, target):
        """Train the model on a batch of data.

        Args:
            data: Input batch for the model.
            target: Expected batch of labels for the data.
        
        Returns:
            Batch loss and predictions.
        """
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()  # Set gradients to zero before starting backpropagation
        y_pred = self.model(data)  # Predict output
        loss = l1(self.model, self.criterion(y_pred, target), self.l1_factor)  # Calculate loss

        # Perform backpropagation
        loss.backward()
        self.optimizer.step()

        if self.metric_fn:
            self.metric_fn(target, y_pred)

        # One Cycle Policy for learning rate
        if not self.lr_schedulers['one_cycle_policy'] is None:
            self.lr_schedulers['one_cycle_policy'].step()

        return loss.item()
    
    def train_epoch(self):
        """Run an epoch of model training."""

        self.model.train()
        pbar = ProgressBar(target=len(self.train_loader), width=8)
        for batch_idx, (data, target) in enumerate(self.train_loader, 0):
            # Train a batch
            loss = self.train_batch(data, target)

            # Update Progress Bar
            pbar_values = [('loss', round(loss, 2))]
            if self.metric_fn:
                pbar_values.append((self.metric, self.metric_val))
            
            pbar.update(batch_idx, values=pbar_values)
            
        # Update training history
        pbar_values = [('loss', round(loss, 2))]
        if self.metric_fn:
            pbar_values.append((self.metric, self.metric_val))
        self.update_training_history(loss)
        pbar.add(1, values=pbar_values)

    
    def train_iterations(self):
        """Train model for the 'self.epochs' number of batches."""

        self.model.train()
        pbar = ProgressBar(target=self.epochs, width=8)
        iterator = InfiniteDataLoader(self.train_loader)
        for iteration in range(self.epochs):
            # Train a batch
            data, target = iterator.get_batch()
            loss = self.train_batch(data, target)

            # Update Progress Bar
            pbar_values = [('loss', round(loss, 2))]
            if self.metric_fn:
                pbar_values.append((self.metric, self.metric_val))
            pbar.update(iteration, values=pbar_values)
            
            # Update training history
            self.update_training_history(loss)
        
        pbar.add(1, values=pbar_values)
    
    def validate(self, verbose=True):
        """Validate an epoch of model training.

        Args:
            verbose: Print validation loss and accuracy.
        """

        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                img_batch = data  # This is done to keep data in CPU
                data, target = data.to(self.device), target.to(self.device)  # Get samples
                output = self.model(data)  # Get trained model output
                val_loss += self.criterion(output, target).item()  # Sum up batch loss

                if self.metric_fn:
                    self.metric_fn(target, output)

        val_loss /= len(self.val_loader.dataset)
        self.val_losses.append(val_loss)

        if self.metric_fn:
            self.val_metric.append(self.metric_val)

        if verbose:
            log = f'Validation set: Average loss: {val_loss:.4f}'
            if not self.metric_val is None:
                log += f', {self.metric}: {self.metric_val:.2f}'
            log += '\n'
            print(log)
    
    def save_checkpoint(self, epoch=None):
        if not self.checkpoint is None:
            metric = None
            if self.checkpoint.monitor == 'train_loss':
                metric = self.train_losses[-1]
            elif self.checkpoint.monitor == 'val_loss':
                metric = self.val_losses[-1]
            elif self.metric_fn:
                if self.checkpoint.monitor.startswith('train_'):
                    metric = self.train_metric[-1]
                else:
                    metric = self.val_metric[-1]
            else:
                print('Invalid metric function, can\'t save checkpoint.')
                return
            
            self.checkpoint(self.model, metric, epoch)
    
    def fit(self):
        """Perform model training."""

        self.reset_history()
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}:')

            # Train an epoch
            self.train_epoch()
            self._reset_metric()
            
            # Validate the model
            if not self.val_loader is None:
                self.validate()
                self._reset_metric()
            
            # Save model checkpoint
            self.save_checkpoint(epoch)

            # Call Step LR
            if not self.lr_schedulers['step_lr'] is None:
                self.lr_schedulers['step_lr'].step()

            # Call Reduce LR on Plateau
            if not self.lr_schedulers['lr_plateau'] is None:
                self.lr_schedulers['lr_plateau'].step(self.val_losses[-1])
