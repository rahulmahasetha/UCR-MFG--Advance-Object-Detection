from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyLoss(nn.Module):
    """Loss for PreGraphUncertaintyHead: Stride-Normalized Gaussian Box NLL + Cls NLL."""

    def __init__(self, strides: List[int] = [8, 16, 32], lambda_cls: float = 1.0):
        super().__init__()
        self.strides = strides
        self.lambda_cls = lambda_cls

    def forward(
        self,
        unc_preds: Dict[str, List[torch.Tensor]],
        reg_preds: List[torch.Tensor],
        cls_targets: List[torch.Tensor],
        reg_targets: List[torch.Tensor],
        pos_masks: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        device = cls_targets[0].device
        total_pos = sum([pos.sum().item() for pos in pos_masks])
        num_pos = max(float(total_pos), 1.0)

        box_nll_losses = []
        cls_nll_losses = []

        log_vars = unc_preds["log_vars"]
        cls_logits = unc_preds["cls_logits"]

        for lvl in range(len(self.strides)):
            s_i = log_vars[lvl].permute(0, 2, 3, 1).reshape(-1, 4)
            logits_i = cls_logits[lvl].permute(0, 2, 3, 1).reshape(-1, cls_logits[lvl].shape[1])
            d_pred = reg_preds[lvl].permute(0, 2, 3, 1).reshape(-1, 4)

            t_reg = reg_targets[lvl].reshape(-1, 4)
            t_cls = cls_targets[lvl].reshape(-1)
            is_pos = pos_masks[lvl].reshape(-1)

            if is_pos.any():
                pos_s = s_i[is_pos].clamp(-10.0, 10.0)
                pos_d_pred = d_pred[is_pos].detach()  # Stop grad into det head
                pos_t_reg = t_reg[is_pos]

                # Stride-normalized coordinate error squared
                sq_err = (pos_t_reg - pos_d_pred) ** 2
                nll_box = 0.5 * torch.exp(-pos_s) * sq_err + 0.5 * pos_s
                box_nll_losses.append(nll_box.sum())

                # Classification NLL on positive ground truth
                pos_logits = logits_i[is_pos]
                pos_labels = t_cls[is_pos]
                nll_cls = F.cross_entropy(pos_logits, pos_labels, reduction="sum")
                cls_nll_losses.append(nll_cls)

        total_box_nll = sum(box_nll_losses) / num_pos if box_nll_losses else torch.tensor(0.0, device=device)
        total_cls_nll = sum(cls_nll_losses) / num_pos if cls_nll_losses else torch.tensor(0.0, device=device)
        l_u = total_box_nll + self.lambda_cls * total_cls_nll

        return {
            "loss_u": l_u,
            "loss_box_nll": total_box_nll,
            "loss_cls_nll": total_cls_nll,
        }

