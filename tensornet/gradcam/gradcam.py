import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class GradCAM:
    """Calculate GradCAM salinecy map.

    *Note*: The current implemenation supports only ResNet models. The class can
    be extended to add support for other models.

    Args:
        model (torch.nn.Module): A model instance.
        layer_name (str): Name of the layer in model for which the
            map will be calculated.
    """

    def __init__(self, model: torch.nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self._target_layer()

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def _target_layer(self):
        layer_num = int(self.layer_name.lstrip('layer'))
        if layer_num == 1:
            self.target_layer = self.model.layer1
        elif layer_num == 2:
            self.target_layer = self.model.layer2
        elif layer_num == 3:
            self.target_layer = self.model.layer3
        elif layer_num == 4:
            self.target_layer = self.model.layer4

    def saliency_map_size(self, *input_size):
        """Returns the shape of the saliency map."""
        device = next(self.model.parameters()).device
        self.model(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def _forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(
        self, input: tuple, class_idx: Optional[int] = None, retain_graph: bool = False
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            input (tuple): Input image with shape of (1, 3, H, W)
            class_idx (:obj:`int`, optional): Class index for calculating GradCAM.
                If not specified, the class index that makes the highest model
                prediction score will be used.

        Returns:
            2-element tuple containing

            - (*torch.tensor*): saliency map of the same spatial dimension with input.
            - (*torch.tensor*): model output.
        """
        return self._forward(input, class_idx, retain_graph)
