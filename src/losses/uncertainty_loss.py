from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


BETA_NLL = 0.5
LOG_VAR_MIN = -6.0


def gaussian_nll(log_var: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Summed heteroscedastic Gaussian NLL over log ltrb distances (a log-normal box model).

    Log space keeps the error scale-free and bounded for untrained heads; the mean is detached so
    variance learning cannot move the boxes.
    """
    # Floor of -6 caps the error weight exp(-log_var) at ~400. At -10 (~22,000x) an over-confident
    # head turned one hard batch into exploding gradients that wrecked the shared backbone (UAVDT ep 55).
    # Soft floor at LOG_VAR_MIN: a hard clamp has zero gradient below it, so an over-confident head stuck
    # there could never become less confident again. softplus keeps a gradient everywhere.
    log_var = (LOG_VAR_MIN + F.softplus(log_var.float() - LOG_VAR_MIN)).clamp(max=10.0)
    sq_err = (target.float().clamp(min=1e-3).log() - pred.detach().float().clamp(min=1e-3).log()) ** 2
    nll = 0.5 * torch.exp(-log_var) * sq_err + 0.5 * log_var
    # beta-NLL (Seitzer et al., ICLR 2022, beta=0.5): weighting by detached sigma^(2*beta) keeps the optimum
    # (sigma^2 = squared error) but shrinks the gradient of over-confident points from ~e^6 to ~e^3.
    return (torch.exp(BETA_NLL * log_var).detach() * nll).sum()


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
