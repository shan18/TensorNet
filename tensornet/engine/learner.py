import math
import torch
import torch.nn.functional as F

from tensornet.engine.ops.regularizer import l1
from tensornet.engine.ops.checkpoint import ModelCheckpoint
from tensornet.data.processing import InfiniteDataLoader
from tensornet.utils.progress_bar import ProgressBar


class Learner:

    def __init__(
        self, model, optimizer, criterion, train_loader, device='cpu',
        epochs=1, val_loader=None, l1_factor=0.0, callbacks=None, metrics=None
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
            metrics (list of str, optional): List of names of the metrics for model
                evaluation. (default: None)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.val_loader = val_loader
        self.l1_factor = l1_factor
        
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
        self.train_metrics = {}  # Change in evaluation metric

        self.val_losses = []  # Change in loss
        self.val_metrics = {}  # Change in evaluation metric

        # Set evaluation metrics
        self.metrics = {}
        if metrics:
            self._setup_metrics(metrics)
    
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
    
    def _accuracy(self, label, prediction):
        """Calculate accuracy.
        
        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        
        Returns:
            accuracy
        """
        self.metrics['accuracy']['sum'] += prediction.eq(
            label.view_as(prediction)
        ).sum().item()
        self.metrics['accuracy']['num_steps'] += len(label)
        self.metrics['accuracy']['value'] = round(
            100 * self.metrics['accuracy']['sum'] / self.metrics['accuracy']['num_steps'], 2
        )
    
    def _pred_label_diff(self, label, prediction, rel=False):
        """Calculate the difference between label and prediction.
        
        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
            rel (bool, optional): If True, return the relative
                difference. (default: False)
        
        Returns:
            Difference between label and prediction
        """
        # For numerical stability
        valid_labels = label > 0.0001
        _label = label[valid_labels]
        _prediction = prediction[valid_labels]
        valid_element_count = _label.size(0)

        if valid_element_count > 0:
            diff = torch.abs(_label - _prediction)
            if rel:
                diff = torch.div(diff, _label)
            
            return diff, valid_element_count

    
    def _rmse(self, label, prediction):
        """Calculate Root Mean Square Error.
        
        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        
        Returns:
            Root Mean Square Error
        """
        diff = self._pred_label_diff(label, prediction)
        rmse = 0
        if not diff is None:
            rmse = math.sqrt(torch.sum(torch.pow(diff[0], 2)) / diff[1])
        
        self.metrics['rmse']['num_steps'] += label.size(0)
        self.metrics['rmse']['sum'] += rmse * label.size(0)
        self.metrics['rmse']['value'] = round(
            self.metrics['rmse']['sum'] / self.metrics['rmse']['num_steps'], 3
        )
    
    def _mae(self, label, prediction):
        """Calculate Mean Average Error.
        
        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        
        Returns:
            Mean Average Error
        """
        diff = self._pred_label_diff(label, prediction)
        mae = 0
        if not diff is None:
            mae = torch.sum(diff[0]).item() / diff[1]
        
        self.metrics['mae']['num_steps'] += label.size(0)
        self.metrics['mae']['sum'] += mae * label.size(0)
        self.metrics['mae']['value'] = round(
            self.metrics['mae']['sum'] / self.metrics['mae']['num_steps'], 3
        )
    
    def _abs_rel(self, label, prediction):
        """Calculate Absolute Relative Error.
        
        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        
        Returns:
            Absolute Relative Error
        """
        diff = self._pred_label_diff(label, prediction, rel=True)
        abs_rel = 0
        if not diff is None:
            abs_rel = torch.sum(diff[0]).item() / diff[1]
        
        self.metrics['abs_rel']['num_steps'] += label.size(0)
        self.metrics['abs_rel']['sum'] += abs_rel * label.size(0)
        self.metrics['abs_rel']['value'] = round(
            self.metrics['abs_rel']['sum'] / self.metrics['abs_rel']['num_steps'], 3
        )
    
    def _setup_metrics(self, metrics):
        """Validate the evaluation metrics passed to the class."""
        for metric in metrics:
            metric_info = {'value': 0, 'sum': 0, 'num_steps': 0}
            if metric == 'accuracy':
                metric_info['func'] = self._accuracy
            elif metric == 'rmse':
                metric_info['func'] = self._rmse
            elif metric == 'mae':
                metric_info['func'] = self._mae
            elif metric == 'abs_rel':
                metric_info['func'] = self._abs_rel
            
            if 'func' in metric_info:
                self.metrics[metric] = metric_info
                self.train_metrics[metric] = []
                self.val_metrics[metric] = []
    
    def _calculate_metrics(self, label, prediction):
        """Update evaluation metric values.
        
        Args:
            label (torch.Tensor): Ground truth.
            prediction (torch.Tensor): Prediction.
        """
        # If predictions are ont-hot encoded
        if label.size() != prediction.size():
            prediction = prediction.argmax(dim=1, keepdim=True) * 1.0
        
        for _, info in self.metrics.items():
            info['func'](label, prediction)
    
    def _reset_metrics(self):
        """Reset metric params."""
        for metric in self.metrics:
            self.metrics[metric]['value'] = 0
            self.metrics[metric]['sum'] = 0
            self.metrics[metric]['num_steps'] = 0
    
    def _get_pbar_values(self, loss):
        pbar_values = [('loss', round(loss, 2))]
        if self.metrics:
            for metric, info in self.metrics.items():
                pbar_values.append((metric, info['value']))
        return pbar_values

    def update_training_history(self, loss):
        """Update the training history."""
        self.train_losses.append(loss)
        for metric in self.metrics:
            self.train_metrics[metric].append(self.metrics[metric]['value'])
    
    def reset_history(self):
        """Reset the training history"""
        self.train_losses = []
        self.val_losses = []
        for metric in self.metrics:
            self.train_metrics[metric] = []
            self.val_metrics[metric] = []
        self._reset_metrics()
    
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

        self._calculate_metrics(target, y_pred)

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
            pbar_values = self._get_pbar_values(loss)
            pbar.update(batch_idx, values=pbar_values)
            
        # Update training history
        self.update_training_history(loss)
        pbar_values = self._get_pbar_values(loss)
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
            pbar_values = self._get_pbar_values(loss)
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
                self._calculate_metrics(target, output)  # Calculate evaluation metrics

        val_loss /= len(self.val_loader.dataset)
        self.val_losses.append(val_loss)

        for metric, info in self.metrics.items():
            self.val_metrics[metric].append(info['value'])

        if verbose:
            log = f'Validation set: Average loss: {val_loss:.4f}'
            for metric, info in self.metrics.items():
                log += f', {metric}: {info["value"]}'
            log += '\n'
            print(log)
    
    def save_checkpoint(self, epoch=None):
        if not self.checkpoint is None:
            metric = None
            params = {}
            if self.checkpoint.monitor == 'train_loss':
                metric = self.train_losses[-1]
            elif self.checkpoint.monitor == 'val_loss':
                metric = self.val_losses[-1]
            elif self.metrics:
                if self.checkpoint.monitor.startswith('train_'):
                    metric = self.train_metrics[self.checkpoint.monitor.split('train_')[-1]][-1]
                else:
                    metric = self.val_metrics[self.checkpoint.monitor.split('val_')[-1]][-1]
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
            self._reset_metrics()
            
            # Validate the model
            if not self.val_loader is None:
                self.validate()
                self._reset_metrics()
            
            # Save model checkpoint
            self.save_checkpoint(epoch)

            # Call Step LR
            if not self.lr_schedulers['step_lr'] is None:
                self.lr_schedulers['step_lr'].step()

            # Call Reduce LR on Plateau
            if not self.lr_schedulers['lr_plateau'] is None:
                self.lr_schedulers['lr_plateau'].step(self.val_losses[-1])
