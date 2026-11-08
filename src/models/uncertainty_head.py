import math
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_batch_mc_generator(base_seed: int, epoch: int, batch_idx: int, t: int, device: torch.device) -> torch.Generator:
    """Derives an independent, reproducible random generator per batch and MC forward pass."""
    seed = (base_seed * 1000003 + epoch * 10007 + batch_idx * 997 + t * 101) & 0x7FFFFFFF
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def apply_reproducible_dropout(x: torch.Tensor, p: float, generator: torch.Generator) -> torch.Tensor:
    """Inverted dropout with explicit generator support."""
    if p <= 0.0:
        return x
    keep_prob = 1.0 - p
    mask = (torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype) < keep_prob)
    return (x * mask) / keep_prob


class PreGraphUncertaintyHead(nn.Module):
    """Contribution 1: Decoupled Pre-Graph Uncertainty Head.
    
    Predicts aleatoric uncertainty Ua and epistemic uncertainty Ue directly from FPN features.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        mc_passes: int = 10,
        sigma_0_sq: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.mc_passes = mc_passes
        self.sigma_0_sq = sigma_0_sq

        # Shared 2-Conv tower
        self.tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.SiLU()
        )

        # Three independent task heads
        self.box_var_head = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.centerness_head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward_single(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single deterministic forward pass returning (log_var, cls_logits, c_pre)."""
        feat = self.tower(f)
        log_var = self.box_var_head(feat)
        cls_logits = self.cls_head(feat)
        c_pre = torch.sigmoid(self.centerness_head(feat))
        return log_var, cls_logits, c_pre

    def compute_aleatoric(
        self,
        log_var: torch.Tensor,
        cls_logits: torch.Tensor,
        c_pre: torch.Tensor
    ) -> torch.Tensor:
        """Computes normalized aleatoric uncertainty Ua in [0, 1]."""
        probs = F.softmax(cls_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
        norm_entropy = entropy / math.log(self.num_classes)

        # Stride-normalized variance average
        mean_var = torch.mean(torch.exp(log_var.clamp(-10.0, 10.0)), dim=1, keepdim=True)
        scaled_var = mean_var / (self.sigma_0_sq + mean_var)

        u_a = 0.5 * norm_entropy + 0.5 * (c_pre * scaled_var)
        return u_a.clamp(0.0, 1.0)

    @torch.no_grad()
    def compute_epistemic(
        self,
        f: torch.Tensor,
        base_seed: int = 42,
        epoch: int = 0,
        batch_idx: int = 0
    ) -> torch.Tensor:
        """Computes epistemic uncertainty Ue via Monte Carlo Inverted Dropout without graph tracking."""
        if not self.training and self.mc_passes <= 1:
            return torch.zeros((f.shape[0], 1, f.shape[2], f.shape[3]), device=f.device, dtype=f.dtype)

        b, c, h, w = f.shape
        device = f.device
        sum_probs = torch.zeros((b, self.num_classes, h, w), device=device, dtype=f.dtype)
        sum_entropy = torch.zeros((b, 1, h, w), device=device, dtype=f.dtype)

        for t in range(self.mc_passes):
            gen = get_batch_mc_generator(base_seed, epoch, batch_idx, t, device)
            f_drop = apply_reproducible_dropout(f, self.dropout_rate, gen)
            feat = self.tower(f_drop)
            logits_t = self.cls_head(feat)
            probs_t = F.softmax(logits_t, dim=1)
            entropy_t = -torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=1, keepdim=True)

            sum_probs += probs_t
            sum_entropy += entropy_t

        mean_probs = sum_probs / float(self.mc_passes)
        mean_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1, keepdim=True)
        expected_entropy = sum_entropy / float(self.mc_passes)

        # Mutual Information
        mutual_info = (mean_entropy - expected_entropy).clamp(min=0.0)
        u_e = mutual_info / math.log(self.num_classes)
        return u_e.clamp(0.0, 1.0)

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        base_seed: int = 42,
        epoch: int = 0,
        batch_idx: int = 0
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """Forward pass for multi-scale pyramid (F1, F2, F3).
        
        Returns:
            u_a_list: Aleatoric uncertainties [Ua_1, Ua_2, Ua_3]
            u_e_list: Epistemic uncertainties [Ue_1, Ue_2, Ue_3]
            preds: Dict containing pre-graph outputs (log_var, cls_logits, c_pre) for loss computation
        """
        u_a_list, u_e_list = [], []
        log_var_list, cls_logits_list, c_pre_list = [], [], []

        for f in features:
            log_var, cls_logits, c_pre = self.forward_single(f)
            # Pass detached tensors to avoid unneeded backprop graphs in Ua and Ue
            u_a = self.compute_aleatoric(log_var.detach(), cls_logits.detach(), c_pre.detach())
            u_e = self.compute_epistemic(f.detach(), base_seed, epoch, batch_idx)

            u_a_list.append(u_a)
            u_e_list.append(u_e)
            log_var_list.append(log_var)
            cls_logits_list.append(cls_logits)
            c_pre_list.append(c_pre)

        preds = {
            "log_vars": log_var_list,
            "cls_logits": cls_logits_list,
            "c_pres": c_pre_list,
        }
        return u_a_list, u_e_list, preds

