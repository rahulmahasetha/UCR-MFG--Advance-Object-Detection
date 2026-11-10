from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def resize_and_pad(
    image: Image.Image,
    target_size: Tuple[int, int] = (1024, 1024)
) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
    """Resizes an image with aspect ratio preservation and pads with 114 (standard gray).
    
    Returns:
        padded_tensor: Float32 tensor [3, target_H, target_W] in [0, 1].
        scale: Resize scale factor.
        padding: (pad_w, pad_h) offsets.
    """
    w, h = image.size
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    image_resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    img_np = np.array(image_resized, dtype=np.float32)

    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    pad_w = (target_w - nw) // 2
    pad_h = (target_h - nh) // 2

    canvas = np.full((target_h, target_w, 3), 114.0, dtype=np.float32)
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw, :] = img_np

    tensor = torch.from_numpy(canvas).permute(2, 0, 1) / 255.0
    return tensor, scale, (pad_w, pad_h)


def transform_boxes(
    boxes: torch.Tensor,
    scale: float,
    padding: Tuple[int, int],
    target_size: Tuple[int, int]
) -> torch.Tensor:
    """Transforms bounding boxes [x1, y1, x2, y2] to target coordinates."""
    if boxes.numel() == 0:
        return boxes
    pad_w, pad_h = padding
    target_h, target_w = target_size
    transformed = boxes.clone()
    transformed[:, [0, 2]] = transformed[:, [0, 2]] * scale + pad_w
    transformed[:, [1, 3]] = transformed[:, [1, 3]] * scale + pad_h
    # Clip to boundary
    transformed[:, [0, 2]] = transformed[:, [0, 2]].clamp(0, target_w)
    transformed[:, [1, 3]] = transformed[:, [1, 3]].clamp(0, target_h)
    return transformed


def transform_ignore_mask(
    mask: np.ndarray,
    scale: float,
    padding: Tuple[int, int],
    target_size: Tuple[int, int]
) -> torch.Tensor:
    """Resizes and pads binary ignore mask to target dimensions."""
    target_h, target_w = target_size
    pad_w, pad_h = padding
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    w, h = mask_pil.size
    nw, nh = int(round(w * scale)), int(round(h * scale))
    mask_resized = np.array(mask_pil.resize((nw, nh), Image.Resampling.NEAREST), dtype=np.uint8)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = mask_resized
    return torch.from_numpy(canvas)

