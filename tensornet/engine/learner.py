import torch
import torch.nn.functional as F

from tensornet.engine.ops.regularizer import l1
from tensornet.data.processing import InfiniteDataLoader
from tensornet.utils.progress_bar import ProgressBar


class Learner:

    def __init__(
        self, model, optimizer, criterion, train_loader, device='cpu',
        epochs=1, val_loader=None, l1_factor=0.0, callbacks=None
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
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.val_loader = val_loader
        self.l1_factor = l1_factor

        self.callbacks = {
            'step_lr': None,
            'lr_plateau': None,
            'one_cycle_policy': None
        }
        if not callbacks is None:
            self._setup_callbacks(callbacks)

        # Training
        self.train_losses = []  # Change in loss
        self.train_accuracies = []  # Change in accuracy
        self.train_correct = 0  # Total number of correctly predicted samples so far
        self.train_processed = 0  # Total number of predicted samples so far

        self.val_losses = []  # Change in loss
        self.val_accuracies = []  # Change in accuracy
    
    def _setup_callbacks(self, callbacks):
        for callback in callbacks:
            if isinstance(callback, torch.optim.lr_scheduler.StepLR):
                self.callbacks['step_lr'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.callbacks['lr_plateau'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.OneCycleLR):
                self.callbacks['one_cycle_policy'] = callback
    
    def update_training_history(self, loss, accuracy):
        """Update the training history."""
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
    
    def reset_history(self):
        """Reset the training history"""
        self.train_losses = []
        self.train_accuracies = []
        self.train_correct = 0
        self.train_processed = 0
        self.val_losses = []
        self.val_accuracies = []
    
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

        pred = y_pred.argmax(dim=1, keepdim=True)
        self.train_correct += pred.eq(target.view_as(pred)).sum().item()
        self.train_processed += len(data)

        # One Cycle Policy for learning rate
        if not self.callbacks['one_cycle_policy'] is None:
            self.callbacks['one_cycle_policy'].step()

        return loss.item()
    
    def train_epoch(self):
        """Run an epoch of model training."""

        self.model.train()
        pbar = ProgressBar(target=len(self.train_loader), width=8)
        for batch_idx, (data, target) in enumerate(self.train_loader, 0):
            # Train a batch
            loss = self.train_batch(data, target)

            # Update Progress Bar
            accuracy = 100 * self.train_correct / self.train_processed
            pbar.update(batch_idx, values=[
                ('loss', round(loss, 2)), ('accuracy', round(accuracy, 2))
            ])
            
        # Update training history
        accuracy = 100 * self.train_correct / self.train_processed
        self.update_training_history(loss, accuracy)
        pbar.add(1, values=[
            ('loss', round(loss, 2)), ('accuracy', round(accuracy, 2))
        ])

    
    def train_iterations(self):
        """Train model for the 'self.epochs' number of batches."""

        self.model.train()
        pbar = ProgressBar(target=self.epochs, width=8)
        iterator = InfiniteDataLoader(self.train_loader)
        correct = 0
        processed = 0
        for iteration in range(self.epochs):
            # Train a batch
            data, target = iterator.get_batch()
            loss = self.train_batch(data, target)

            # Update Progress Bar
            accuracy = 100 * self.train_correct / self.train_processed
            pbar.update(iteration, values=[
                ('loss', round(loss, 2)), ('accuracy', round(accuracy, 2))
            ])
            
            # Update training history
            self.update_training_history(loss, accuracy)
        
        pbar.add(1, values=[
            ('loss', round(loss, 2)), ('accuracy', round(accuracy, 2))
        ])
    
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
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
                result = pred.eq(target.view_as(pred))
                correct += result.sum().item()

        val_loss /= len(self.val_loader.dataset)
        val_accuracy = 100. * correct / len(self.val_loader.dataset)
        
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)

        if verbose:
            print(
                f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n'
            )
    
    def fit(self):
        """Perform model training."""

        self.reset_history()
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}:')

            # Train an epoch
            self.train_epoch()

            # Call Step LR
            if not self.callbacks['step_lr'] is None:
                self.callbacks['step_lr'].step()
            
            # Validate the model
            if not self.val_loader is None:
                self.validate()

            # Call Reduce LR on Plateau
            if not self.callbacks['lr_plateau'] is None:
                self.callbacks['lr_plateau'].step(self.val_losses[-1])
