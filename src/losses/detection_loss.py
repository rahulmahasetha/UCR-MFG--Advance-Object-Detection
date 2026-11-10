from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.uncertainty_loss import gaussian_nll


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25
) -> torch.Tensor:
    """Sigmoid focal loss on classification logits."""
    # logits: [N, C], targets: [N] (0..C-1 for classes, C for background, -1 for ignore)
    valid_mask = targets >= 0
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits = logits[valid_mask]
    targets = targets[valid_mask]
    num_classes = logits.shape[1]

    # One-hot encoding (background class C becomes all zeros)
    target_one_hot = torch.zeros_like(logits)
    fg_mask = targets < num_classes
    target_one_hot[fg_mask, targets[fg_mask]] = 1.0

    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, target_one_hot, reduction="none")
    p_t = p * target_one_hot + (1 - p) * (1 - target_one_hot)
    loss = ce_loss * ((1 - p_t) ** gamma)

    alpha_t = alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)
    loss = alpha_t * loss
    return loss.sum()


def ciou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """Computes Complete IoU (CIoU) loss for positive predicted boxes."""
    # Boxes: [N, 4] in [x1, y1, x2, y2]
    if pred_boxes.numel() == 0:
        return torch.empty((0,), device=pred_boxes.device)

    x1_p, y1_p, x2_p, y2_p = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    x1_t, y1_t, x2_t, y2_t = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

    w_p, h_p = (x2_p - x1_p).clamp(min=eps), (y2_p - y1_p).clamp(min=eps)
    w_t, h_t = (x2_t - x1_t).clamp(min=eps), (y2_t - y1_t).clamp(min=eps)

    # Intersection
    inter_x1 = torch.max(x1_p, x1_t)
    inter_y1 = torch.max(y1_p, y1_t)
    inter_x2 = torch.min(x2_p, x2_t)
    inter_y2 = torch.min(y2_p, y2_t)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    area_p = w_p * h_p
    area_t = w_t * h_t
    union_area = area_p + area_t - inter_area + eps
    iou = inter_area / union_area

    # Enclosing box
    c_x1 = torch.min(x1_p, x1_t)
    c_y1 = torch.min(y1_p, y1_t)
    c_x2 = torch.max(x2_p, x2_t)
    c_y2 = torch.max(y2_p, y2_t)
    c_diag_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Center distance
    ctr_p_x, ctr_p_y = (x1_p + x2_p) / 2.0, (y1_p + y2_p) / 2.0
    ctr_t_x, ctr_t_y = (x1_t + x2_t) / 2.0, (y1_t + y2_t) / 2.0
    rho_sq = (ctr_p_x - ctr_t_x) ** 2 + (ctr_p_y - ctr_t_y) ** 2

    # Aspect ratio penalty v
    v = (4.0 / (torch.pi ** 2)) * torch.pow(torch.atan(w_t / h_t) - torch.atan(w_p / h_p), 2)
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + eps)

    ciou = iou - (rho_sq / c_diag_sq) - alpha * v
    return 1.0 - ciou.clamp(min=-1.0, max=1.0)


class FCOSDetectionLoss(nn.Module):
    """FCOS detection loss (focal + centerness-weighted CIoU + centerness BCE) and the box-variance NLL
    of the probabilistic head, reported separately as `loss_var`."""

    def __init__(self, strides: List[int] = [8, 16, 32]):
        super().__init__()
        self.strides = strides

    def forward(
        self,
        head: Dict[str, List[torch.Tensor]],
        cls_targets: List[torch.Tensor],
        reg_targets: List[torch.Tensor],
        cen_targets: List[torch.Tensor],
        pos_masks: List[torch.Tensor],
        locations: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = head["cls"][0].device
        num_pos = max(float(sum(p.sum() for p in pos_masks)), 1.0)
        loss_cls = torch.zeros((), device=device)
        loss_box = torch.zeros((), device=device)
        loss_cen = torch.zeros((), device=device)
        loss_var = torch.zeros((), device=device)

        for lvl, stride in enumerate(self.strides):
            batch_size, num_classes = head["cls"][lvl].shape[:2]
            pred_cls = head["cls"][lvl].permute(0, 2, 3, 1).reshape(-1, num_classes).float()
            loss_cls = loss_cls + focal_loss(pred_cls, cls_targets[lvl].reshape(-1))

            pos = pos_masks[lvl].reshape(-1)
            if not pos.any():
                continue
            pred_reg = head["reg"][lvl].permute(0, 2, 3, 1).reshape(-1, 4)[pos].float()
            pred_var = head["log_var"][lvl].permute(0, 2, 3, 1).reshape(-1, 4)[pos]
            pred_cen = head["cen"][lvl].permute(0, 2, 3, 1).reshape(-1)[pos].float()
            t_reg = reg_targets[lvl].reshape(-1, 4)[pos]
            t_cen = cen_targets[lvl].reshape(-1)[pos]

            loc = locations[lvl].repeat(batch_size, 1)[pos]
            ciou = ciou_loss(_decode(loc, pred_reg, stride), _decode(loc, t_reg, stride))
            loss_box = loss_box + (ciou * t_cen).sum()
            loss_cen = loss_cen + F.binary_cross_entropy_with_logits(pred_cen, t_cen, reduction="sum")
            loss_var = loss_var + gaussian_nll(pred_var, pred_reg, t_reg)

        loss_cls, loss_box, loss_cen, loss_var = (x / num_pos for x in (loss_cls, loss_box, loss_cen, loss_var))
        return {
            "loss_det": loss_cls + loss_box + loss_cen,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_cen": loss_cen,
            "loss_var": loss_var,
        }


def _decode(loc: torch.Tensor, ltrb: torch.Tensor, stride: int) -> torch.Tensor:
    ltrb = ltrb * stride
    return torch.cat([loc - ltrb[:, :2], loc + ltrb[:, 2:]], dim=1)
