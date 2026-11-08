from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Unified FCOS Detection Loss: Focal Classification + CIoU Regression + BCE Centerness."""

    def __init__(self, strides: List[int] = [8, 16, 32]):
        super().__init__()
        self.strides = strides

    def forward(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        cen_preds: List[torch.Tensor],
        cls_targets: List[torch.Tensor],
        reg_targets: List[torch.Tensor],
        cen_targets: List[torch.Tensor],
        pos_masks: List[torch.Tensor],
        locations: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        device = cls_preds[0].device
        total_pos = sum([pos.sum().item() for pos in pos_masks])
        num_pos = max(float(total_pos), 1.0)

        loss_cls_list, loss_box_list, loss_cen_list = [], [], []

        for lvl, stride in enumerate(self.strides):
            pred_cls = cls_preds[lvl].permute(0, 2, 3, 1).reshape(-1, cls_preds[lvl].shape[1])
            pred_reg = reg_preds[lvl].permute(0, 2, 3, 1).reshape(-1, 4)
            pred_cen = cen_preds[lvl].permute(0, 2, 3, 1).reshape(-1)

            t_cls = cls_targets[lvl].reshape(-1)
            t_reg = reg_targets[lvl].reshape(-1, 4)
            t_cen = cen_targets[lvl].reshape(-1)
            is_pos = pos_masks[lvl].reshape(-1)

            # 1. Focal loss
            loss_cls_list.append(focal_loss(pred_cls, t_cls))

            # 2. CIoU & Centerness on positive locations
            if is_pos.any():
                pos_pred_reg = pred_reg[is_pos]
                pos_t_reg = t_reg[is_pos]
                pos_t_cen = t_cen[is_pos]
                pos_pred_cen = pred_cen[is_pos]

                # Decode boxes from normalized targets
                loc_repeated = locations[lvl].repeat(cls_preds[lvl].shape[0], 1)[is_pos]
                pred_x1 = loc_repeated[:, 0] - pos_pred_reg[:, 0] * stride
                pred_y1 = loc_repeated[:, 1] - pos_pred_reg[:, 1] * stride
                pred_x2 = loc_repeated[:, 0] + pos_pred_reg[:, 2] * stride
                pred_y2 = loc_repeated[:, 1] + pos_pred_reg[:, 3] * stride
                decoded_preds = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

                tgt_x1 = loc_repeated[:, 0] - pos_t_reg[:, 0] * stride
                tgt_y1 = loc_repeated[:, 1] - pos_t_reg[:, 1] * stride
                tgt_x2 = loc_repeated[:, 0] + pos_t_reg[:, 2] * stride
                tgt_y2 = loc_repeated[:, 1] + pos_t_reg[:, 3] * stride
                decoded_targets = torch.stack([tgt_x1, tgt_y1, tgt_x2, tgt_y2], dim=-1)

                ciou_raw = ciou_loss(decoded_preds, decoded_targets)
                loss_box = (ciou_raw * pos_t_cen).sum() / (pos_t_cen.sum() + 1e-7)
                loss_box_list.append(loss_box * pos_t_cen.sum())

                loss_cen = F.binary_cross_entropy_with_logits(pos_pred_cen, pos_t_cen, reduction="sum")
                loss_cen_list.append(loss_cen)

        total_cls_loss = sum(loss_cls_list) / num_pos
        total_box_loss = sum(loss_box_list) / num_pos if loss_box_list else torch.tensor(0.0, device=device)
        total_cen_loss = sum(loss_cen_list) / num_pos if loss_cen_list else torch.tensor(0.0, device=device)

        l_det = total_cls_loss + total_box_loss + total_cen_loss

        return {
            "loss_det": l_det,
            "loss_cls": total_cls_loss,
            "loss_box": total_box_loss,
            "loss_cen": total_cen_loss,
        }

