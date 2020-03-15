import torch
from torchvision import transforms


def transformations(horizontal_flip=0.0, vertical_flip=0.0, rotation=0.0, random_erasing=0.0):
    """Create data transformations
    
    Args:
        horizontal_flip: Probability of an image being horizontally flipped.
            Defaults to 0.
        vertical_flip: Probability of an image being vertically flipped.
            Defaults to 0.
        rotation: Angle of rotation for image augmentation.
            Defaults to 0.
        random_erasing: Probability that random erase will be performed.
            Defaults to 0.
    
    Returns:
        Transform object containing defined data transformations.
    """

    transforms_list = []

    if horizontal_flip > 0:  # Horizontal Flip
        transforms_list += [transforms.RandomHorizontalFlip(horizontal_flip)]
    if vertical_flip > 0:  # Vertical Flip
        transforms_list += [transforms.RandomVerticalFlip(vertical_flip)]
    if rotation > 0:  # Rotate image
        transforms_list += [transforms.RandomRotation(rotation, fill=1)]
    
    transforms_list += [
        # convert the data to torch.FloatTensor
        # with values within the range [0.0 ,1.0]
        transforms.ToTensor(),

        # normalize the data with mean and standard deviation to keep values in range [-1, 1]
        # since there are 3 channels for each image,
        # we have to specify mean and std for each channel
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    if random_erasing > 0:
        transforms_list += [transforms.RandomErasing(random_erasing)]
    
    return transforms.Compose(transforms_list)


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
