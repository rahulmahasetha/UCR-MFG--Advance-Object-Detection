from typing import Tuple, Dict, Any
import random
import torch
import numpy as np


class TrainAugmenter:
    """Stochastic training augmentations strictly disjoint from evaluation corruptions."""

    def __init__(self, flip_p: float = 0.5, jitter_p: float = 0.5):
        self.flip_p = flip_p
        self.jitter_p = jitter_p

    def __call__(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        ignore_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Horizontal Flip
        if random.random() < self.flip_p:
            image = torch.flip(image, dims=[2])
            ignore_mask = torch.flip(ignore_mask, dims=[1])
            if boxes.numel() > 0:
                w = image.shape[2]
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1

        # Color Jitter (Brightness & Contrast)
        if random.random() < self.jitter_p:
            brightness_factor = 1.0 + random.uniform(-0.2, 0.2)
            contrast_factor = 1.0 + random.uniform(-0.2, 0.2)
            image = torch.clamp(image * brightness_factor, 0.0, 1.0)
            mean = image.mean(dim=[-2, -1], keepdim=True)
            image = torch.clamp((image - mean) * contrast_factor + mean, 0.0, 1.0)

        return image, boxes, ignore_mask

