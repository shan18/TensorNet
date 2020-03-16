import cv2
import torch

import matplotlib.pyplot as plt

from .gradcam import GradCAM
from .gradcam_pp import GradCAMPP
from data.utils import to_numpy


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.

    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Returns:

        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """

    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


def plot_cam(data, plot_path):
    """Display data.

    Args:
        data: List of images, heatmaps and result.
        plot_path: Complete path for saving the plot.
    """

    # Initialize plot
    fig, axs = plt.subplots(len(data), 5, figsize=(10, 10))

    for idx, cam in enumerate(data):
        axs[idx][0].axis('off')
        axs[idx][1].axis('off')
        axs[idx][2].axis('off')
        axs[idx][3].axis('off')
        axs[idx][4].axis('off')

        if idx == 0:
            axs[idx][0].set_title('Input')
            axs[idx][1].set_title('GradCAM Heatmap')
            axs[idx][2].set_title('GradCAM Result')
            axs[idx][3].set_title('GradCAM++ Heatmap')
            axs[idx][4].set_title('GradCAM++ Result')

        # Plot image
        axs[idx][0].imshow(cam['image'])
        axs[idx][1].imshow(cam['cam_heatmap'])
        axs[idx][2].imshow(cam['cam_result'])
        axs[idx][3].imshow(cam['campp_heatmap'])
        axs[idx][4].imshow(cam['campp_result'])
    
    # Set spacing
    fig.tight_layout()

    # Save image
    fig.savefig(f'{plot_path}', bbox_inches='tight')


def display_cam(model, device, samples, unnormalize, plot_path):
    """ Given a set of images, display CAM for each of them.
        Also save the generated image.

    Args:
        model: Trained model.
        samples: List of images in torch.Tensor format.
        plot_path: Path to store the generated image.
    """

    # Initialize GradCAM
    gradcam = GradCAM(model, 'layer4')
    gradcam_pp = GradCAMPP(model, 'layer4')

    images = []
    for sample in samples:
        norm_image = sample.clone().unsqueeze_(0).to(device)
        image = unnormalize(sample)

        mask, _ = gradcam(norm_image)
        heatmap, result = visualize_cam(mask, norm_image)

        mask_pp, _ = gradcam_pp(norm_image)
        heatmap_pp, result_pp = visualize_cam(mask_pp, norm_image)

        images.append({
            'image': image,
            'cam_heatmap': to_numpy(heatmap),
            'cam_result': to_numpy(result),
            'campp_heatmap': to_numpy(heatmap_pp),
            'campp_result': to_numpy(result_pp)
        })

    plot_cam(images, plot_path)
