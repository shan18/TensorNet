""" This file contains common-utility functions
    that could be needed by any dataset.
"""


def unnormalize(image, mean, std):
    """Un-normalize a given image.
    
    Args:
        image: A numpy array with shape (H, W, 3).
        mean: Mean value. It can be a single value or
            a tuple with 3 values (one for each channel).
        std: Standard deviation value. It can be a single value or
            a tuple with 3 values (one for each channel).
    """

    return image * std + mean
