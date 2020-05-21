from torch import nn
from torch.nn import functional as F
import torch


class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        """Dice Loss.

        Args:
            smooth (float, optional): Smoothing value. A larger
                smooth value (also known as Laplace smooth, or
                Additive smooth) can be used to avoid overfitting.
                (default: 1)
        """
        super(DiceLoss, self).__init__()

        self.smooth = 1

    def forward(self, input, target):
        """Calculate Dice Loss.

        Args:
            input (torch.Tensor): Model predictions.
            target (torch.Tensor): Target values.
        
        Returns:
            dice loss
        """
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = (input_flat * target_flat).sum()
        union = input_flat.sum() + target_flat.sum()

        return 1 - ((2. * intersection + self.smooth) / (union + self.smooth))
