import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt

from tensornet.gradcam.gradcam import GradCAM
from tensornet.gradcam.gradcam_pp import GradCAMPP
from tensornet.data.utils import to_numpy, unnormalize
from typing import Tuple, List, Dict, Union, Optional


def visualize_cam(mask: torch.Tensor, img: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor]:
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.

    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Returns:
        2-element tuple containing

        - (*torch.tensor*): heatmap img shape of (3, H, W)
        - (*torch.tensor*): synthesized GradCAM result of same shape with heatmap.
    """

    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


class GradCAMView:
    """Create GradCAM and GradCAM++.

    *Note*: The current implemenation of `GradCAM` and `GradCAM++` supports only ResNet
    models. The class can be extended to add support for other models.

    Args:
        model (torch.nn.Module): Trained model.
        layers (list): List of layers to show GradCAM on.
        device (:obj:`str` or :obj:`torch.device`): GPU or CPU.
        mean (:obj:`float` or :obj:`tuple`): Mean of the dataset.
        std (:obj:`float` or :obj:`tuple`): Standard Deviation of the dataset.
    """

    def __init__(
        self, model: torch.nn.Module, layers: List[str], device: Union[str, torch.device],
        mean: Union[float, tuple], std: Union[float, tuple]
    ):
        self.model = model
        self.layers = layers
        self.device = device
        self.mean = mean
        self.std = std

        self._gradcam()
        self._gradcam_pp()

        print('Mode set to GradCAM.')
        self.grad = self.gradcam.copy()

        self.views = []

    def _gradcam(self):
        """Initialize GradCAM instance."""
        self.gradcam = {}
        for layer in self.layers:
            self.gradcam[layer] = GradCAM(self.model, layer)

    def _gradcam_pp(self):
        """Initialize GradCAM++ instance."""
        self.gradcam_pp = {}
        for layer in self.layers:
            self.gradcam_pp[layer] = GradCAMPP(self.model, layer)

    def switch_mode(self):
        """Switch between GradCAM and GradCAM++."""
        if self.grad == self.gradcam:
            print('Mode switched to GradCAM++.')
            self.grad = self.gradcam_pp.copy()
        else:
            print('Mode switched to GradCAM.')
            self.grad = self.gradcam.copy()

    def _cam_image(
        self, norm_image: torch.Tensor, class_idx: Optional[int] = None
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Get CAM for an image.

        Args:
            norm_image (torch.Tensor): Normalized image.
            class_idx (:obj:`int`, optional): Class index for calculating GradCAM.
                If not specified, the class index that makes the highest model
                prediction score will be used.

        Returns:
            Dictionary containing unnormalized image, heatmap and CAM result.
        """
        image = unnormalize(norm_image, self.mean, self.std)  # Unnormalized image
        norm_image_cuda = norm_image.clone().unsqueeze_(0).to(self.device)
        heatmap, result = {}, {}
        for layer, gc in self.gradcam.items():
            mask, _ = gc(norm_image_cuda, class_idx=class_idx)
            cam_heatmap, cam_result = visualize_cam(
                mask,
                image.clone().unsqueeze_(0).to(self.device)
            )
            heatmap[layer], result[layer] = to_numpy(cam_heatmap), to_numpy(cam_result)
        return {
            'image': to_numpy(image),
            'heatmap': heatmap,
            'result': result
        }

    def cam(self, norm_img_class_list: List[Union[Dict[str, Union[torch.Tensor, int]], torch.Tensor]]):
        """Get CAM for a list of images.

        Args:
            norm_img_class_list (list): List of dictionaries or list of images.
                If dict, each dict contains keys 'image' and 'class'
                having values 'normalized_image' and 'class_idx' respectively.
                class_idx is optional. If class_idx is not given then the
                model prediction will be used and the parameter should just be
                a list of images. Each image should be of type torch.Tensor
        """
        for norm_image_class in norm_img_class_list:
            class_idx = None
            norm_image = norm_image_class
            if type(norm_image_class) == dict:
                class_idx, norm_image = norm_image_class['class'], norm_image_class['image']
            self.views.append(self._cam_image(norm_image, class_idx=class_idx))

    def __call__(
        self, norm_img_class_list: List[Union[Dict[str, Union[torch.Tensor, int]], torch.Tensor]]
    ) -> List[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
        """Get GradCAM for a list of images.

        Args:
            norm_img_class_list (list): List of dictionaries or list of images.
                If dict, each dict contains keys 'image' and 'class'
                having values 'normalized_image' and 'class_idx' respectively.
                class_idx is optional. If class_idx is not given then the
                model prediction will be used and the parameter should just be
                a list of images. Each image should be of type torch.Tensor
        """
        self.cam(norm_img_class_list)
        return self.views
