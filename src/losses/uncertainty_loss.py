from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_nll(log_var: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Summed heteroscedastic Gaussian NLL over log ltrb distances (a log-normal box model).

    Log space keeps the error scale-free and bounded for untrained heads; the mean is detached so
    variance learning cannot move the boxes.
    """
    log_var = log_var.float().clamp(-10.0, 10.0)
    sq_err = (target.float().clamp(min=1e-3).log() - pred.detach().float().clamp(min=1e-3).log()) ** 2
    return (0.5 * torch.exp(-log_var) * sq_err + 0.5 * log_var).sum()


class UncertaintyLoss(nn.Module):
    """Trains the pre-graph uncertainty head: box-variance NLL plus classification NLL on positives."""

    def __init__(self, lambda_cls: float = 1.0):
        super().__init__()
        self.lambda_cls = lambda_cls

    def forward(
        self,
        unc_preds: Dict[str, List[torch.Tensor]],
        reg_preds: List[torch.Tensor],
        cls_targets: List[torch.Tensor],
        reg_targets: List[torch.Tensor],
        pos_masks: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = reg_preds[0].device
        num_pos = max(float(sum(p.sum() for p in pos_masks)), 1.0)
        box_nll = torch.zeros((), device=device)
        cls_nll = torch.zeros((), device=device)

        for log_var, logits, reg, t_cls, t_reg, pos in zip(
            unc_preds["log_vars"], unc_preds["cls_logits"], reg_preds, cls_targets, reg_targets, pos_masks
        ):
            pos = pos.reshape(-1)
            if not pos.any():
                continue
            box_nll = box_nll + gaussian_nll(
                log_var.permute(0, 2, 3, 1).reshape(-1, 4)[pos],
                reg.permute(0, 2, 3, 1).reshape(-1, 4)[pos],
                t_reg.reshape(-1, 4)[pos],
            )
            pos_logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])[pos]
            cls_nll = cls_nll + F.cross_entropy(pos_logits.float(), t_cls.reshape(-1)[pos], reduction="sum")

        box_nll, cls_nll = box_nll / num_pos, cls_nll / num_pos
        return {"loss_u": box_nll + self.lambda_cls * cls_nll, "loss_box_nll": box_nll, "loss_cls_nll": cls_nll}
