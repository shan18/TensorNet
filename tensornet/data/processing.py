import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensor


class Transformations:
    """ Wrapper class to pass on albumentaions transforms into PyTorch. """

    def __init__(self, horizontal_flip_prob=0.0, vertical_flip_prob=0.0, rotate_degree=0.0, cutout=0.0):
        """Create data transformation pipeline
        
        Args:
            horizontal_flip_prob: Probability of an image being horizontally flipped.
                Defaults to 0.
            vertical_flip_prob: Probability of an image being vertically flipped.
                Defaults to 0.
            rotate_degree: Angle of rotation for image augmentation.
                Defaults to 0.
            cutout: Probability that cutout will be performed.
                Defaults to 0.
        """

        self.transform = A.Compose([
            A.HorizontalFlip(p=horizontal_flip_prob),  # Horizontal Flip
            A.VerticalFlip(p=vertical_flip_prob),  # Vertical Flip
            A.Rotate(limit=rotate_degree),  # Rotate image

            # CutOut
            A.CoarseDropout(
                p=cutout, max_holes=1, fill_value=(0.5 * 255, 0.5 * 255, 0.5 * 255),
                max_height=16, max_width=16, min_height=1, min_width=1
            ),

            # normalize the data with mean and standard deviation to keep values in range [-1, 1]
            # since there are 3 channels for each image,
            # we have to specify mean and std for each channel
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),

            # convert the data to torch.FloatTensor
            # with values within the range [0.0 ,1.0]
            ToTensor(),
        ])
    
    def __call__(self, image):
        """Process and image through the data transformation pipeline.

        Args:
            image: Image to process.
        
        Returns:
            Transformed image.
        """

        image = np.array(image)
        image = self.transform(image=image)['image']
        return image


def data_loader(data, shuffle=True, batch_size=1, num_workers=1, cuda=False):
    """Create data loader

    Args:
        data: Downloaded dataset.
        batch_size: Number of images to considered in each batch.
        num_workers: How many subprocesses to use for data loading.
        cuda: True is GPU is available.
    
    Returns:
        DataLoader instance.
    """

    loader_args = {
        'shuffle': shuffle,
        'batch_size': batch_size
    }

    # If GPU exists
    if cuda:
        loader_args['num_workers'] = num_workers
        loader_args['pin_memory'] = True
    
    return torch.utils.data.DataLoader(data, **loader_args)
