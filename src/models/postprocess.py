from typing import Dict, List, Sequence

import torch
from torchvision.ops import batched_nms


@torch.no_grad()
def postprocess(
    outputs: Dict,
    locations: Sequence[torch.Tensor],
    strides: Sequence[int],
    image_size: Sequence[int],
    score_thresh: float = 0.05,
    pre_nms_topk: int = 1000,
    nms_iou: float = 0.6,
    max_dets: int = 500,
) -> List[Dict[str, torch.Tensor]]:
    """Decodes detections with per-detection uncertainty: box std (px), aleatoric Ua and epistemic Ue."""
    head = outputs["head"]
    batch_size = head["cls"][0].shape[0]
    h_img, w_img = image_size
    results = []

    for b in range(batch_size):
        parts = {k: [] for k in ("boxes", "scores", "labels", "box_std", "u_a", "u_e")}
        for lvl, stride in enumerate(strides):
            cls = head["cls"][lvl][b].float().sigmoid().flatten(1).T
            cen = head["cen"][lvl][b].float().sigmoid().flatten()
            scores = (cls * cen.unsqueeze(1)).sqrt().flatten()
            cand = torch.nonzero(scores > score_thresh).squeeze(1)
            if cand.numel() > pre_nms_topk:
                cand = cand[scores[cand].topk(pre_nms_topk).indices]
            loc_idx, labels = cand // cls.shape[1], cand % cls.shape[1]

            reg = head["reg"][lvl][b].float().flatten(1).T[loc_idx] * stride
            loc = locations[lvl][loc_idx]
            parts["boxes"].append(torch.cat([loc - reg[:, :2], loc + reg[:, 2:]], dim=1))
            parts["scores"].append(scores[cand])
            parts["labels"].append(labels)
            log_var = head["log_var"][lvl][b].float().flatten(1).T[loc_idx]
            parts["box_std"].append(reg * (torch.exp(0.5 * log_var) - 1.0))
            parts["u_a"].append(outputs["u_a"][lvl][b].flatten()[loc_idx])
            parts["u_e"].append(outputs["u_e"][lvl][b].flatten()[loc_idx])

        det = {k: torch.cat(v) for k, v in parts.items()}
        det["boxes"][:, 0::2] = det["boxes"][:, 0::2].clamp(0, w_img)
        det["boxes"][:, 1::2] = det["boxes"][:, 1::2].clamp(0, h_img)
        keep = batched_nms(det["boxes"], det["scores"], det["labels"], nms_iou)[:max_dets]
        results.append({k: v[keep] for k, v in det.items()})
    return results
