import os
import matplotlib.pyplot as plt
from typing import Union, List, Dict, Tuple, Optional


def plot_metric(
    data: Union[List[float], Dict[str, List[float]]],
    metric: str, title: str = None, size: Tuple[int] = (7, 5),
    legend_font: int = 15, legend_loc: str = 'lower right'
):
    """Plot accuracy graph or loss graph.

    Args:
        data (:obj:`list` or :obj:`dict`): If only single plot then this is a list, else
            for multiple plots this is a dict with keys containing the plot name and values
            being a list of points to plot.
        metric (str): Metric name which is to be plotted. Can be either
            loss or accuracy.
        title (:obj:`str`, optional): Title of the plot, if no title given then it is
            determined from the x and y label.
        size (:obj:`tuple`, optional): Size of the plot. (default: **'(7, 5)'**)
        legend_loc (:obj:`str`, optional): Location of the legend box in the plot.
            No legend will be plotted if there is only a single plot.
            (default: *'lower right'*)
        legend_font (:obj:`int`, optional): Font size of the legend (default: *'15'*)
    """

    single_plot = True
    if type(data) == dict:
        single_plot = False

    # Initialize a figure
    fig = plt.figure(figsize=size)

    # Plot data
    if single_plot:
        plt.plot(data)
    else:
        plots = []
        for value in data.values():
            plots.append(plt.plot(value)[0])

    # Set plot title
    if title is None:
        title = f'{metric} Change'
    plt.title(title)

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    if not single_plot:  # Set legend
        plt.legend(
            tuple(plots), tuple(data.keys()),
            loc=legend_loc,
            shadow=True,
            prop={'size': legend_font}
        )

    # Save plot
    fig.savefig(f'{"_".join(title.split()).lower()}.png')


def plot_predictions(
    data: List[dict], classes: Union[List[str], Tuple[str]],
    plot_title: str, plot_path: str
):
    """Display data.

    Args:
        data (list): List of images, model predictions and ground truths.
            Images should be numpy arrays.
        classes (:obj:`list` or :obj:`tuple`): List of classes in the dataset.
        plot_title (str): Title for the plot.
        plot_path (str): Complete path for saving the plot.
    """

    # Initialize plot
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle(plot_title)

    for idx, result in enumerate(data):

        # If 25 samples have been stored, break out of loop
        if idx > 24:
            break

        label = result['label'].item()
        prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}')
        axs[row_count][idx % 5].imshow(result['image'])

    # Set spacing
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    # Save image
    fig.savefig(f'{plot_path}', bbox_inches='tight')


def save_and_show_result(
    classes: Union[List[str], Tuple[str]], correct_pred: Optional[List[dict]] = None,
    incorrect_pred: Optional[List[dict]] = None, path: Optional[str] = None
):
    """Display network predictions.

    Args:
        classes (:obj:`list` or :obj:`tuple`): List of classes in the dataset.
        correct_pred (:obj:`list`, optional): Contains correct model predictions and labels.
        incorrect_pred (:obj:`list`, optional): Contains incorrect model predictions and labels.
        path (:obj:`str`, optional): Path where the results will be saved.
    """

    # Create directories for saving predictions
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'predictions'
        )
    if not os.path.exists(path):
        os.makedirs(path)

    if not correct_pred is None:  # Plot correct predicitons
        plot_predictions(
            correct_pred, classes, 'Correct Predictions', f'{path}/correct_predictions.png'
        )

    if not incorrect_pred is None:  # Plot incorrect predicitons
        plot_predictions(
            incorrect_pred, classes, '\nIncorrect Predictions', f'{path}/incorrect_predictions.png'
        )
