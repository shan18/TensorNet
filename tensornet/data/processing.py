import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensor


class Transformations:
    """ Wrapper class to pass on albumentaions transforms into PyTorch. """

    def __init__(
        self, horizontal_flip_prob=0.0, vertical_flip_prob=0.0, gaussian_blur_prob=0.0,
        rotate_degree=0.0, cutout=0.0, cutout_height=0, cutout_width=0,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), train=True
    ):
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
            cutout_height: Max height of the cutout box.
                Defaults to 0.
            cutout_width: Max width of the cutout box.
                Defaults to 0.
            mean: Mean value. Defaults to 0.5 for each channel.
            std: Standard deviation value. Defaults to 0.5 for each channel.
        """
        transforms_list = []

        if train:
            if horizontal_flip_prob > 0:  # Horizontal Flip
                transforms_list += [A.HorizontalFlip(p=horizontal_flip_prob)]
            if vertical_flip_prob > 0:  # Vertical Flip
                transforms_list += [A.VerticalFlip(p=vertical_flip_prob)]
            if gaussian_blur_prob > 0:  # Patch Gaussian Augmentation
                transforms_list += [A.GaussianBlur(p=gaussian_blur_prob)]
            if rotate_degree > 0:  # Rotate image
                transforms_list += [A.Rotate(limit=rotate_degree)]
            if cutout > 0:  # CutOut
                transforms_list += [A.CoarseDropout(
                    p=cutout, max_holes=1, fill_value=tuple([x * 255.0 for x in mean]),
                    max_height=cutout_height, max_width=cutout_width, min_height=1, min_width=1
                )]
        
        transforms_list += [
            # normalize the data with mean and standard deviation to keep values in range [-1, 1]
            # since there are 3 channels for each image,
            # we have to specify mean and std for each channel
            A.Normalize(mean=mean, std=std, always_apply=True),
            
            # convert the data to torch.FloatTensor
            # with values within the range [0.0 ,1.0]
            ToTensor()
        ]

        self.transform = A.Compose(transforms_list)
    
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
