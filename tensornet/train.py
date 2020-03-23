import torch.nn.functional as F
from tqdm import tqdm

from tensornet.model.utils.regularizer import l1


def train(
    model, loader, device, optimizer, criterion,
    losses=None, accuracies=None, track='epoch', l1_factor=0.0
):
    """Train the model.

    Args:
        model (torch.nn.Module): Model Instance.
        device (str or torch.device): Device where the data
            will be loaded.
        loader (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim): Optimizer for the model.
        criterion (torch.nn): Loss Function.
        losses (list, optional): List containing the change in loss.
            (default: None)
        accuracies (list, optional): List containing the change in accuracy.
            (default: None)
        track (str, optional): Can be set to either 'epoch' or 'batch' and will
            store the changes in loss and accuracy for each batch
            or the entire epoch respectively. (default: 'epoch')
        l1_factor (float, optional): L1 regularization factor. (default: 0)
    """

    model.train()
    pbar = tqdm(loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar, 0):
        # Get samples
        data, target = data.to(device), target.to(device)

        # Set gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Predict output
        y_pred = model(data)

        # Calculate loss
        loss = l1(model, criterion(y_pred, target), l1_factor)

        # Perform backpropagation
        loss.backward()
        optimizer.step()

        # Update Progress Bar
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        if track == 'batch':  # Store loss and accuracy
            batch_accuracy = 100 * correct / processed
            if not losses is None:
                losses.append(loss.item())
            if not accuracies is None:
                accuracies.append(batch_accuracy)

        pbar.set_description(
            desc=f'Loss={loss.item():0.2f} Batch_ID={batch_idx} Accuracy={(100 * correct / processed):.2f}'
        )

    if track == 'epoch':  # Store loss and accuracy
        accuracy = 100 * correct / processed
        if not losses is None:
            losses.append(loss.item())
        if not accuracies is None:
            accuracies.append(accuracy)
