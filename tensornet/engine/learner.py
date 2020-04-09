import torch
import torch.nn.functional as F
from tqdm import tqdm

from tensornet.engine.ops.regularizer import l1


class Learner:

    def __init__(
        self, model, optimizer, criterion, train_loader, device='cpu',
        epochs=1, val_loader=None, l1_factor=0.0, callbacks=None, track='epoch'
    ):
        """Train and validate the model.

        Args:
            model (torch.nn.Module): Model Instance.
            optimizer (torch.optim): Optimizer for the model.
            criterion (torch.nn): Loss Function.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            device (str or torch.device, optional): Device where the data
                will be loaded. (default='cpu')
            epochs (int, optional): Numbers of epochs to train the model. (default: 1)
            val_loader (torch.utils.data.DataLoader, optional): Validation data
                loader. (default: None)
            l1_factor (float, optional): L1 regularization factor. (default: 0)
            callbacks (list, optional): List of callbacks to be used during training.
                (default: None)
            track (str, optional): Can be set to either 'epoch' or 'batch' and will
                store the changes in loss and accuracy for each batch
                or the entire epoch respectively. (default: 'epoch')
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.val_loader = val_loader
        self.l1_factor = l1_factor
        self.track = track

        self.callbacks = {
            'step_lr': None,
            'lr_plateau': None,
            'one_cycle_policy': None
        }
        if not callbacks is None:
            self._setup_callbacks(callbacks)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def _setup_callbacks(self, callbacks):
        for callback in callbacks:
            if isinstance(callback, torch.optim.lr_scheduler.StepLR):
                self.callbacks['step_lr'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.callbacks['lr_plateau'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.OneCycleLR):
                self.callbacks['one_cycle_policy'] = callback
    
    def train_epoch(self):
        """Run an epoch of model training."""

        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar, 0):
            data, target = data.to(self.device), target.to(self.device)  # Get samples
            self.optimizer.zero_grad()  # Set gradients to zero before starting backpropagation
            y_pred = self.model(data)  # Predict output
            loss = l1(self.model, self.criterion(y_pred, target), self.l1_factor)  # Calculate loss

            # Perform backpropagation
            loss.backward()
            self.optimizer.step()

            # One Cycle Policy for learning rate
            if not self.callbacks['one_cycle_policy'] is None:
                self.callbacks['one_cycle_policy'].step()

            # Update Progress Bar
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            if self.track == 'batch':  # Store batch-level loss and accuracy
                batch_accuracy = 100 * correct / processed
                if not sel.train_losses is None:
                    self.train_losses.append(loss.item())
                if not self.train_accuracies is None:
                    self.train_accuracies.append(batch_accuracy)

            pbar.set_description(
                desc=f'Loss={loss.item():0.2f} Batch_ID={batch_idx} Accuracy={(100 * correct / processed):.2f}'
            )

        if self.track == 'epoch':  # Store epoch-level loss and accuracy
            accuracy = 100 * correct / processed
            if not self.train_losses is None:
                self.train_losses.append(loss.item())
            if not self.train_accuracies is None:
                self.train_accuracies.append(accuracy)
    
    def validate_epoch(self):
        """Validate an epoch of model training."""

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
        
        if not self.val_losses is None:
            self.val_losses.append(val_loss)
        if not self.val_accuracies is None:
            self.val_accuracies.append(val_accuracy)

        print(
            f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(self.val_loader.dataset)} ({val_accuracy:.2f}%)\n'
        )
    
    def fit(self):
        """Perform model training."""
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}:')

            # Train an epoch
            self.train_epoch()

            # Call Step LR
            if not self.callbacks['step_lr'] is None:
                self.callbacks['step_lr'].step()
            
            # Validate an epoch
            self.validate_epoch()

            # Call Reduce LR on Plateau
            if not self.callbacks['lr_plateau'] is None:
                self.callbacks['lr_plateau'].step(self.val_losses[-1])
