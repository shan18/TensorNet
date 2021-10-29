import torch
import torch.nn as nn

from .utils.summary import summary as model_summary
from tensornet.engine.learner import Learner
from typing import Tuple


class BaseModel(nn.Module):
    """This is the parent class for all the models that are to be
    created using ``TensorNet``."""

    def __init__(self):
        """This function instantiates all the model layers."""
        super(BaseModel, self).__init__()
        self.learner = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This function defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input.

        Returns:
            (*torch.Tensor*): Model output.
        """
        raise NotImplementedError

    def summary(self, input_size: Tuple[int]):
        """Generates model summary.

        Args:
            input_size (tuple): Size of input to the model.
        """
        model_summary(self, input_size)

    def create_learner(
        self, train_loader, optimizer, criterion, device='cpu',
        epochs=1, l1_factor=0.0, val_loader=None, callbacks=None, metrics=None,
        activate_loss_logits=False, record_train=True
    ):
        """Create Learner object.

        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim): Optimizer for the model.
            criterion (torch.nn): Loss Function.
            device (:obj:`str` or :obj:`torch.device`): Device where the data will be loaded.
            epochs (:obj:`int`, optional): Numbers of epochs to train the model. (default: 1)
            l1_factor (:obj:`float`, optional): L1 regularization factor. (default: 0)
            val_loader (:obj:`torch.utils.data.DataLoader`, optional): Validation data loader.
            callbacks (:obj:`list`, optional): List of callbacks to be used during training.
            track (:obj:`str`, optional): Can be set to either `'epoch'` or `'batch'` and will store the
                changes in loss and accuracy for each batch or the entire epoch respectively.
                (default: *'epoch'*)
            metrics (:obj:`list`, optional): List of names of the metrics for model evaluation.
        """
        self.learner = Learner(
            train_loader, optimizer, criterion, device=device, epochs=epochs,
            val_loader=val_loader, l1_factor=l1_factor, callbacks=callbacks, metrics=metrics,
            activate_loss_logits=activate_loss_logits, record_train=record_train
        )
        self.learner.set_model(self)

    def set_learner(self, learner: Learner):
        """Assign a learner object to the model.

        Args:
            learner (:obj:`Learner`): Learner object.
        """
        self.learner = learner
        self.learner.set_model(self)

    def fit(
        self, train_loader, optimizer, criterion, device='cpu', epochs=1,
        l1_factor=0.0, val_loader=None, callbacks=None, metrics=None,
        activate_loss_logits=False, record_train=True, start_epoch=1, verbose=True,
    ):
        """Train the model.

        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim): Optimizer for the model.
            criterion (torch.nn): Loss Function.
            device (:obj:`str` or :obj:`torch.device`): Device where the data will be loaded.
            epochs (:obj:`int`, optional): Numbers of epochs to train the model. (default: 1)
            l1_factor (:obj:`float`, optional): L1 regularization factor. (default: 0)
            val_loader (:obj:`torch.utils.data.DataLoader`, optional): Validation data loader.
            callbacks (:obj:`list`, optional): List of callbacks to be used during training.
            track (:obj:`str`, optional): Can be set to either `'epoch'` or `'batch'` and will store the
                changes in loss and accuracy for each batch or the entire epoch respectively.
                (default: *'epoch'*)
            metrics (:obj:`list`, optional): List of names of the metrics for model evaluation.
            record_train (:obj:`bool`, optional): If False, metrics will be calculated only
                during validation. (default: True)
            activate_loss_logits (:obj:`bool`, optional): If True, the logits will first pass
                through the `activate_logits` function before going to the criterion.
                (default: False)
            start_epoch (:obj:`int`, optional): Starting epoch number to display during training.
                (default: 1)
            verbose (:obj:`bool`, optional): Print loss and metrics. (default: True)
        """

        # Create learner object
        self.create_learner(
            train_loader, optimizer, criterion, device=device, epochs=epochs, l1_factor=l1_factor,
            val_loader=val_loader, callbacks=callbacks, metrics=metrics,
            activate_loss_logits=activate_loss_logits, record_train=record_train,
        )

        # Train Model
        self.learner.fit(start_epoch=start_epoch, epochs=epochs, verbose=verbose)

    def rfit(self, start_epoch=1, epochs=None, verbose=True):
        if self.learner is None:
            raise ValueError('No learner initialized.')

        self.learner.fit(start_epoch=start_epoch, epochs=epochs, reset=False, verbose=verbose)

    def evaluate(self, loader, verbose=True, log_message='Evaluation'):
        """Evaluate the model on a custom data loader.

        Args:
            loader (torch.utils.data.DataLoader): Data loader.
            verbose (:obj:`bool`, optional): Print loss and metrics. (default: True)
            log_message (str): Prefix for the logs which are printed at the end.

        Returns:
            loss and metric values
        """

        if self.learner is None:
            raise ValueError('Cannot evaluate without a learner. Create and assign a learner object first.')

        return self.learner.evaluate(loader, verbose=verbose, log_message=log_message)

    def save(self, filepath: str, **kwargs):
        """Save the model.

        Args:
            filepath (str): File in which the model will be saved.
            **kwargs: Additional parameters to save with the model.
        """
        if self.learner is None:
            raise ValueError('Cannot save un-trained model.')

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.learner.optimizer.state_dict(),
            **kwargs
        }, filepath)

    def load(self, filepath: str) -> dict:
        """Load the model and return the additional parameters saved in
        in the checkpoint file.

        Args:
            filepath (str): File in which the model is be saved.

        Returns:
            (*dict*): Parameters saved inside the checkpoint file.
        """
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        return {
            k: v for k, v in checkpoint.items() if k != 'model_state_dict'
        }
