from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.graph_ops import Routing
from src.models.postprocess import postprocess
from src.models.uncertainty_head import mc_generator
from src.utils.metrics import compute_dece, compute_map, uncertainty_auroc


class LocationCache:
    """Pixel centers of every head location, cached per feature-map shape."""

    def __init__(self, head):
        self.head = head
        self._cache: Dict[Tuple, List[torch.Tensor]] = {}

    def __call__(self, cls_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        shapes = tuple(tuple(m.shape[-2:]) for m in cls_maps)
        key = (shapes, str(cls_maps[0].device))
        if key not in self._cache:
            self._cache[key] = self.head.get_grid_locations(list(shapes), cls_maps[0].device)
        return self._cache[key]


def assign_targets(model, locations: List[torch.Tensor], batch: Dict[str, Any], device) -> Dict[str, List]:
    boxes = [b.to(device) for b in batch["boxes"]]
    labels = [l.to(device) for l in batch["labels"]]
    cls_t, reg_t, cen_t, pos = model.det_head.assign_targets(locations, boxes, labels, batch["ignore_masks"].to(device))
    return {"cls_targets": cls_t, "reg_targets": reg_t, "cen_targets": cen_t, "pos_masks": pos}


def measure_benefit(loss_det: torch.Tensor, routing: Routing, scale: float = 1.0) -> torch.Tensor:
    """b_ij = -dL_det/dm_ij: first-order decrease of the detection loss from strengthening edge ij."""
    (grad,) = torch.autograd.grad(loss_det * scale, routing.mask, retain_graph=True)
    return -grad.detach()


@torch.no_grad()
def predict(
    model,
    loader,
    device,
    infer_cfg: Dict[str, Any],
    label_map: Optional[Dict[int, int]] = None,
    max_batches: Optional[int] = None,
    seed: int = 0,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    model.eval()
    locations = LocationCache(model.det_head)
    preds, gts = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["images"].to(device)
        outputs = model(images, generator=mc_generator(seed, 0, i, device))
        dets = postprocess(outputs, locations(outputs["head"]["cls"]), model.strides, images.shape[-2:], **infer_cfg)
        for b, det in enumerate(dets):
            det = {k: v.cpu().numpy() for k, v in det.items()}
            if label_map is not None:
                keep = np.isin(det["labels"], list(label_map))
                det = {k: v[keep] for k, v in det.items()}
                det["labels"] = np.array([label_map[int(l)] for l in det["labels"]], dtype=np.int64)
            preds.append(det)
            gts.append({
                "boxes": batch["boxes"][b].numpy(),
                "labels": batch["labels"][b].numpy(),
                "ignore_mask": batch["ignore_masks"][b].numpy(),
            })
    return preds, gts


def evaluate(model, loader, device, infer_cfg: Dict[str, Any], num_classes: int, **kwargs) -> Dict[str, Any]:
    preds, gts = predict(model, loader, device, infer_cfg, **kwargs)
    metrics = compute_map(preds, gts, num_classes)
    metrics["D-ECE"] = compute_dece(preds, gts, num_classes=num_classes)
    metrics["AUROC"] = uncertainty_auroc(preds, gts)
    metrics["num_images"] = len(preds)
    return metrics
