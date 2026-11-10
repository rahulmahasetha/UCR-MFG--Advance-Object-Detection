from typing import List, Sequence

import torch
import torch.nn as nn

FUSION_MODES = ("gate", "fixed", "add")


class UncertaintyFusionGate(nn.Module):
    """C4: Uncertainty-aware fusion Z = lambda * F + (1 - lambda) * H.

    lambda in (0, 1)^C is predicted per location from [F, H, Ua, Ue]. The gate starts near the visual
    features (lambda ~ 0.88) so freshly enabled graph features cannot disrupt detection.
    `fixed` (lambda = 0.5) and `add` (F + H) are ablation baselines.
    """

    def __init__(self, channels: int = 256, mode: str = "gate", init_visual_bias: float = 2.0):
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"fusion mode must be one of {FUSION_MODES}, got {mode!r}")
        self.mode = mode
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, channels, 1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.constant_(self.gate_conv[-1].bias, init_visual_bias)

    def forward(
        self,
        f_list: Sequence[torch.Tensor],
        h_list: Sequence[torch.Tensor],
        ua_list: Sequence[torch.Tensor],
        ue_list: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        if self.mode == "add":
            return [f + h for f, h in zip(f_list, h_list)]
        if self.mode == "fixed":
            return [0.5 * (f + h) for f, h in zip(f_list, h_list)]
        fused = []
        for f, h, ua, ue in zip(f_list, h_list, ua_list, ue_list):
            gate = torch.sigmoid(self.gate_conv(torch.cat([f, h, ua.to(f.dtype), ue.to(f.dtype)], dim=1)))
            fused.append(gate * f + (1.0 - gate) * h)
        return fused
