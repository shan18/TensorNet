import torch
from typing import Tuple


def set_seed(seed: int, cuda: bool):
    """Set seed to make the results reproducible.

    Args:
        seed (int): Random seed value.
        cuda (bool): Whether CUDA is available.
    """
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def initialize_cuda(seed: int) -> Tuple[bool, torch.device]:
    """Check if GPU is availabe and set seed.

    Args:
        seed (int): Random seed value.

    Returns:
        2-element tuple containing

        - (*bool*): if cuda is available
        - (*torch.device*): device name
    """

    # Check CUDA availability
    cuda = torch.cuda.is_available()
    print('GPU Available?', cuda)

    # Initialize seed
    set_seed(seed, cuda)

    # Set device
    device = torch.device("cuda" if cuda else "cpu")

    return cuda, device
