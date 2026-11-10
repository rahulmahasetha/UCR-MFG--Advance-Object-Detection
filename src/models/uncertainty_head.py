import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def mc_generator(base_seed: int, epoch: int, batch_idx: int, device: torch.device) -> torch.Generator:
    """Reproducible generator for one batch's Monte Carlo dropout passes."""
    seed = (base_seed * 1000003 + epoch * 10007 + batch_idx * 997) & 0x7FFFFFFF
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def inverted_dropout(x: torch.Tensor, p: float, generator: torch.Generator) -> torch.Tensor:
    if p <= 0.0:
        return x
    keep = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype) >= p
    return x * keep / (1.0 - p)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits.float(), dim=1)
    return -(log_p.exp() * log_p).sum(dim=1, keepdim=True)


class PreGraphUncertaintyHead(nn.Module):
    """C1: Dual predictive uncertainty estimated from pyramid features before graph reasoning.

    With T Monte Carlo dropout passes the predictive entropy splits into
    aleatoric = E_t[H(p_t)] and epistemic = H(E_t[p_t]) - E_t[H(p_t)] (mutual information).
    Ua blends the aleatoric class entropy with a centerness-weighted box variance; both maps lie in [0, 1].
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        mc_passes: int = 10,
        sigma_0_sq: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.mc_passes = mc_passes
        self.sigma_0_sq = sigma_0_sq
        self.log_classes = math.log(num_classes)
        self.tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
        )
        self.box_var_head = nn.Conv2d(in_channels, 4, 1)
        self.cls_head = nn.Conv2d(in_channels, num_classes, 1)
        self.centerness_head = nn.Conv2d(in_channels, 1, 1)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        generator: torch.Generator = None,
        compute_epistemic: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        u_a, u_e = [], []
        preds = {"log_vars": [], "cls_logits": [], "c_pres": []}
        use_mc = compute_epistemic and self.mc_passes > 1

        for f in features:
            feat = self.tower(f)
            log_var = self.box_var_head(feat)
            cls_logits = self.cls_head(feat)
            c_pre = torch.sigmoid(self.centerness_head(feat))

            with torch.no_grad():
                if use_mc:
                    aleatoric_cls, epistemic = self._monte_carlo(f.detach(), generator)
                else:
                    aleatoric_cls, epistemic = entropy(cls_logits) / self.log_classes, torch.zeros_like(c_pre)
                var = log_var.float().clamp(-10.0, 10.0).exp().mean(dim=1, keepdim=True)
                box_term = c_pre.float() * var / (self.sigma_0_sq + var)
                u_a.append((0.5 * aleatoric_cls + 0.5 * box_term).clamp(0.0, 1.0))
                u_e.append(epistemic)

            preds["log_vars"].append(log_var)
            preds["cls_logits"].append(cls_logits)
            preds["c_pres"].append(c_pre)
        return u_a, u_e, preds

    def _monte_carlo(self, f: torch.Tensor, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_probs = 0.0
        mean_entropy = 0.0
        for _ in range(self.mc_passes):
            logits = self.cls_head(self.tower(inverted_dropout(f, self.dropout_rate, generator)))
            mean_probs = mean_probs + F.softmax(logits.float(), dim=1) / self.mc_passes
            mean_entropy = mean_entropy + entropy(logits) / self.mc_passes
        total = -(mean_probs * torch.log(mean_probs.clamp(min=1e-12))).sum(dim=1, keepdim=True)
        epistemic = (total - mean_entropy).clamp(min=0.0)
        return mean_entropy / self.log_classes, (epistemic / self.log_classes).clamp(0.0, 1.0)
