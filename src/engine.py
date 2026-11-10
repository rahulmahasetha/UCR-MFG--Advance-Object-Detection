from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.graph_ops import Routing
from src.models.postprocess import postprocess
from src.models.uncertainty_head import mc_generator
from src.data.corruptions import corrupt
from src.utils.metrics import compute_dece, compute_map, ignored_flags, occlusion_breakdown, uncertainty_auroc


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
    corruption: Optional[Tuple[str, int]] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    """Runs inference and returns per-image detections and ground truth (numpy), optionally on corrupted input."""
    model.eval()
    locations = LocationCache(model.det_head)
    preds, gts = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["images"]
        if corruption is not None:
            images = corrupt(images, corruption[0], corruption[1], seed=seed * 1000003 + i)
        images = images.to(device)
        outputs = model(images, generator=mc_generator(seed, 0, i, device))
        dets = postprocess(outputs, locations(outputs["head"]["cls"]), model.strides, images.shape[-2:], **infer_cfg)
        for b, det in enumerate(dets):
            det = {k: v.cpu().numpy() for k, v in det.items()}
            if label_map is not None:
                keep = np.isin(det["labels"], list(label_map))
                det = {k: v[keep] for k, v in det.items()}
                det["labels"] = np.array([label_map[int(l)] for l in det["labels"]], dtype=np.int64)
            preds.append(det)
            gt = {
                "image_id": batch["image_ids"][b],
                "boxes": batch["boxes"][b].numpy(),
                "labels": batch["labels"][b].numpy(),
                "ignore_mask": batch["ignore_masks"][b].numpy(),
            }
            if "occlusion" in batch:
                gt["occlusion"] = batch["occlusion"][b].numpy()
            if "attributes" in batch:
                gt["attributes"] = batch["attributes"][b]
            preds[-1]["image_id"] = gt["image_id"]
            preds[-1]["center_ignored"] = ignored_flags(preds[-1], gt)
            gts.append(gt)
    return preds, gts


def summarize(preds: List[Dict[str, np.ndarray]], gts: List[Dict[str, Any]], num_classes: int) -> Dict[str, Any]:
    """All paper metrics for one evaluation: accuracy, calibration, uncertainty AUROC, occlusion and condition subsets."""
    metrics = compute_map(preds, gts, num_classes)
    metrics["D-ECE"] = compute_dece(preds, gts, num_classes=num_classes)
    metrics["AUROC"] = uncertainty_auroc(preds, gts)
    metrics["occlusion"] = occlusion_breakdown(preds, gts, num_classes)
    metrics["conditions"] = condition_breakdown(preds, gts, num_classes)
    metrics["num_images"] = len(preds)
    return metrics


def condition_breakdown(preds, gts, num_classes: int) -> Dict[str, Dict[str, Any]]:
    """mAP / AP50 per UAVDT weather and altitude condition (images grouped by sequence attributes)."""
    if not all("attributes" in gt for gt in gts):
        return {}
    out = {}
    for group in ("weather", "altitude"):
        values = sorted({v for gt in gts for v in gt["attributes"].get(group, [])})
        for value in values:
            idx = [i for i, gt in enumerate(gts) if value in gt["attributes"].get(group, [])]
            m = compute_map([preds[i] for i in idx], [gts[i] for i in idx], num_classes)
            out[f"{group}:{value}"] = {"mAP": m["mAP"], "AP50": m["AP50"], "num_images": len(idx)}
    return out


def evaluate(model, loader, device, infer_cfg: Dict[str, Any], num_classes: int, **kwargs) -> Dict[str, Any]:
    preds, gts = predict(model, loader, device, infer_cfg, **kwargs)
    metrics = compute_map(preds, gts, num_classes)
    metrics["D-ECE"] = compute_dece(preds, gts, num_classes=num_classes)
    metrics["num_images"] = len(preds)
    return metrics
