from typing import List
import torch
import torch.nn as nn


class UncertaintyFusionGate(nn.Module):
    """Contribution 4: Uncertainty-Aware Feature Fusion Gate.
    
    Dynamically balances visual features F and relational graph features H via a 
    channel-wise gating mechanism conditioned on predictive uncertainties (Ua, Ue).
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels
        # Input: [F (256), H (256), Ua (1), Ue (1)] -> 514 channels
        in_dim = channels * 2 + 2

        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_dim, channels, kernel_size=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        f_list: List[torch.Tensor],
        h_list: List[torch.Tensor],
        ua_list: List[torch.Tensor],
        ue_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Fuses visual and graph feature pyramids level by level.
        
        Returns:
            z_list: Fused feature pyramid [Z1, Z2, Z3] with C = 256.
        """
        z_list = []
        for f, h, ua, ue in zip(f_list, h_list, ua_list, ue_list):
            concat_in = torch.cat([f, h, ua, ue], dim=1)  # [B, 514, H, W]
            gate = self.gate_conv(concat_in)             # [B, 256, H, W] in (0, 1)
            z = gate * f + (1.0 - gate) * h              # Fused feature map
            z_list.append(z)
        return z_list

