import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def class_level_accuracy(model, loader, device, classes):
    """Print test accuracy for each class in dataset.

    Args:
        model: Model instance.
        loader: Data loader.
        device: Device where data will be loaded.
        classes: List of classes in the dataset.
    """

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader, 0):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


def plot_metric(data, metric):
    """Plot accuracy graph or loss graph.

    Args:
        data: If only single plot then this is a list, else
            for multiple plots this is a dict with keys containing
            the plot name and values being a list of points to plot.
        metric: Metric name which is to be plotted. Can be either
            loss or accuracy.
    """

    single_plot = True
    if type(data) == dict:
        single_plot = False
    
    # Initialize a figure
    fig = plt.figure(figsize=(7, 5))

    # Plot data
    if single_plot:
        plt.plot(data)
    else:
        plots = []
        for value in data.values():
            plots.append(plt.plot(value)[0])

    # Set plot title
    plt.title(f'{metric} Change')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    if not single_plot: # Set legend
        location = 'upper' if metric == 'Loss' else 'lower'
        plt.legend(
            tuple(plots), tuple(data.keys()),
            loc=f'{location} right',
            shadow=True,
            prop={'size': 15}
        )

    # Save plot
    fig.savefig(f'{metric.lower()}_change.png')


def plot_predictions(data, classes, plot_title, plot_path):
    """Display data.

    Args:
        data: List of images, model predictions and ground truths.
        classes: List of classes in the dataset.
        plot_title: Title for the plot.
        plot_path: Complete path for saving the plot.
    """

    # Initialize plot
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle(plot_title)

    for idx, result in enumerate(data):

        # If 25 samples have been stored, break out of loop
        if idx > 24:
            break
        
        rgb_image = np.transpose(result['image'], (1, 2, 0)) / 2 + 0.5
        label = result['label'].item()
        prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}')
        axs[row_count][idx % 5].imshow(rgb_image)
    
    # Set spacing
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    # Save image
    fig.savefig(f'{plot_path}', bbox_inches='tight')


def save_and_show_result(correct_pred, incorrect_pred, classes):
    """Display network predictions.

    Args:
        correct_pred: Contains correct model predictions and labels.
        incorrect_pred: Contains incorrect model predictions and labels.
        classes: List of classes in the dataset.
    """

    # Create directories for saving predictions
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'predictions'
    )
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Plot correct predicitons
    plot_predictions(
        correct_pred, classes, 'Correct Predictions', f'{path}/correct_predictions.png'
    )

    # Plot incorrect predicitons
    plot_predictions(
        incorrect_pred, classes, '\nIncorrect Predictions', f'{path}/incorrect_predictions.png'
    )
