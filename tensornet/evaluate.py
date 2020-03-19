import torch
import torch.nn.functional as F


def evaluate(
    model, loader, device, criterion, losses=None, accuracies=None,
    correct_samples=None, incorrect_samples=None, sample_count=25,
    last_epoch=False
):
    """Train the model.

    Args:
        model: Model instance.
        loader: Validation data loader.
        device: Device where the data will be loaded.
        criterion: Loss function.
        losses: List containing the change in loss.
            Default is None.
        accuracies: List containing the change in accuracy.
            Default is None.
        correct_samples: List containing correctly predicted samples.
            Default is None.
        incorrect_samples: List containing incorrectly predicted samples.
            Default is None.
        sample_count: Total number of predictions to store from each correct
            and incorrect samples. Default is 25.
    """

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            img_batch = data  # This is done to keep data in CPU
            data, target = data.to(device), target.to(device)  # Get samples
            output = model(data)  # Get trained model output
            val_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            result = pred.eq(target.view_as(pred))

            # Save correct and incorrect samples
            if last_epoch:
                for i in range(len(list(result))):
                    if list(result)[i]:
                        if not correct_samples is None and len(correct_samples) < sample_count:
                            correct_samples.append({
                                'prediction': list(pred)[i],
                                'label': list(target.view_as(pred))[i],
                                'image': img_batch[i]
                            })
                    else:
                        if not incorrect_samples is None and len(incorrect_samples) < sample_count:
                            incorrect_samples.append({
                                'prediction': list(pred)[i],
                                'label': list(target.view_as(pred))[i],
                                'image': img_batch[i]
                            })

            correct += result.sum().item()

    val_loss /= len(loader.dataset)
    val_accuracy = 100. * correct / len(loader.dataset)
    
    if not losses is None:
        losses.append(val_loss)
    if not accuracies is None:
        accuracies.append(val_accuracy)

    print(
        f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({val_accuracy:.2f}%)\n'
    )
