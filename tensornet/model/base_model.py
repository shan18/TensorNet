import torch.nn as nn
import torchsummary

from tensornet.engine.learner import Learner


class BaseModel(nn.Module):

    def __init__(self):
        """This function instantiates all the model layers."""
        super(BaseModel, self).__init__()
        self.learner = None
    
    def forward(self, x):
        """This function defines the forward pass of the model.

        Args:
            x: Input.
        
        Returns:
            Model output.
        """
        raise NotImplementedError

    def summary(self, input_size):
        """Generates model summary.

        Args:
            input_size (tuple): Size of input to the model.
        """
        torchsummary.summary(self, input_size)

    def fit(
        self, train_loader, optimizer, criterion, device='cpu',
        epochs=1, l1_factor=0.0, val_loader=None, callbacks=None
    ):
        """Train the model.

        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim): Optimizer for the model.
            criterion (torch.nn): Loss Function.
            device (str or torch.device): Device where the data
                will be loaded.
            epochs (int, optional): Numbers of epochs to train the model. (default: 1)
            l1_factor (float, optional): L1 regularization factor. (default: 0)
            val_loader (torch.utils.data.DataLoader, optional): Validation data
                loader. (default: None)
            callbacks (list, optional): List of callbacks to be used during training.
                (default: None)
            track (str, optional): Can be set to either 'epoch' or 'batch' and will
                store the changes in loss and accuracy for each batch
                or the entire epoch respectively. (default: 'epoch')
        """
        self.learner = Learner(
            self, optimizer, criterion, train_loader, device=device, epochs=epochs,
            val_loader=val_loader, l1_factor=l1_factor, callbacks=callbacks
        )
        self.learner.fit()
