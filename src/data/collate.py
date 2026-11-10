from typing import List, Dict, Any
import torch


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of sample dictionaries into a batched representation."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    ignore_masks = torch.stack([b["ignore_mask"] for b in batch], dim=0)

    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    orig_sizes = [b["orig_size"] for b in batch]
    scale_factors = [b["scale_factor"] for b in batch]

    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "ignore_masks": ignore_masks,
        "image_ids": image_ids,
        "orig_sizes": orig_sizes,
        "scale_factors": scale_factors,
    }

