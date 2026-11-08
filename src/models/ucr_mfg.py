from typing import Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn

from src.models.backbone import ResNetFPNBackbone
from src.models.uncertainty_head import PreGraphUncertaintyHead
from src.models.graph_builder import MultiScaleGraphBuilder
from src.models.router import UCRouter
from src.models.message_passing import BenefitAwareMPNN
from src.models.fusion import UncertaintyFusionGate
from src.models.detection_head import FCOSHead


class UCRMFG(nn.Module):
    """UCR-MFG: Uncertainty-Conditioned Relational Routing for Multi-Scale Fuzzy Graph Object Detection."""

    def __init__(
        self,
        num_classes: int = 10,
        fpn_channels: int = 256,
        graph_dim: int = 64,
        strides: List[int] = [8, 16, 32],
        dropout_rate: float = 0.2,
        mc_passes: int = 10,
        pretrained: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.current_stage = 4  # Default to full system; configured during staged training

        # 1. Primary Backbone & FPN
        self.backbone = ResNetFPNBackbone(pretrained=pretrained, out_channels=fpn_channels)

        # 2. Contribution 1: Decoupled Pre-Graph Uncertainty Head
        self.unc_head = PreGraphUncertaintyHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            mc_passes=mc_passes
        )

        # 3. Multi-Scale Graph Builder
        self.graph_builder = MultiScaleGraphBuilder(
            in_channels=fpn_channels,
            graph_dim=graph_dim
        )

        # 4. Contribution 2: Uncertainty-Conditioned Relational Router
        self.router = UCRouter(
            input_dim=140,
            hidden_dim=graph_dim
        )

        # 5. Contribution 3: Benefit-Aware Multi-Scale Message Passing
        self.mpnn = BenefitAwareMPNN(
            graph_dim=graph_dim,
            out_channels=fpn_channels
        )

        # 6. Contribution 4: Uncertainty-Aware Feature Fusion Gate
        self.fusion_gate = UncertaintyFusionGate(channels=fpn_channels)

        # 7. Downstream FCOS Detection Head
        self.det_head = FCOSHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            strides=strides
        )

    def set_curriculum_stage(self, stage: int) -> None:
        """Sets the active curriculum training stage (1, 2, 3, or 4)."""
        self.current_stage = stage

    def forward(
        self,
        images: torch.Tensor,
        base_seed: int = 42,
        epoch: int = 0,
        batch_idx: int = 0
    ) -> Dict[str, Any]:
        """Full end-to-end forward pass through UCR-MFG."""
        batch_size = images.shape[0]

        # 1. Backbone forward: F1, F2, F3
        features = self.backbone(images)  # Tuple of [B, 256, H_k, W_k]
        spatial_shapes = [(f.shape[2], f.shape[3]) for f in features]

        # 2. Uncertainty Head forward
        # Epistemic uncertainty via MC passes is only needed when graph/router are active (Stage >= 2)
        need_epistemic = (self.current_stage >= 2)
        u_a_list, u_e_list, unc_preds = self.unc_head(
            features, base_seed, epoch, batch_idx, compute_epistemic=need_epistemic
        )

        # In Stage 1, bypass graph modules directly to detection head
        if self.current_stage == 1:
            cls_preds, reg_preds, cen_preds = self.det_head(list(features))
            return {
                "cls_preds": cls_preds,
                "reg_preds": reg_preds,
                "cen_preds": cen_preds,
                "features": features,
                "u_a": u_a_list,
                "u_e": u_e_list,
                "unc_preds": unc_preds,
                "graph_data": None,
                "router_data": None,
                "stage": 1,
            }

        # 3. Candidate Graph Construction
        graph_data = self.graph_builder(features)

        # In Stage 2, graph regularization is active, router is bypassed
        if self.current_stage == 2:
            cls_preds, reg_preds, cen_preds = self.det_head(list(features))
            return {
                "cls_preds": cls_preds,
                "reg_preds": reg_preds,
                "cen_preds": cen_preds,
                "features": features,
                "u_a": u_a_list,
                "u_e": u_e_list,
                "unc_preds": unc_preds,
                "graph_data": graph_data,
                "router_data": None,
                "stage": 2,
            }

        # 4. Uncertainty-Conditioned Relational Routing
        router_outputs = []
        h_pyramids = []

        for b in range(batch_size):
            g_b = graph_data[b]

            # Flatten uncertainties for this sample
            ua_flat = torch.cat([u_a_list[i][b, 0].reshape(-1) for i in range(3)], dim=0)
            ue_flat = torch.cat([u_e_list[i][b, 0].reshape(-1) for i in range(3)], dim=0)

            # Route edges
            r_out = self.router(g_b, ua_flat, ue_flat)
            router_outputs.append(r_out)

            # 5. Benefit-Aware Message Passing
            h_b = self.mpnn(
                nodes=g_b["nodes"],
                edge_index=r_out["edge_index"],
                m_st=r_out["m_st"],
                s_ij=r_out["s_ij"],
                ue_flat=ue_flat,
                scale_splits=g_b["scale_splits"],
                spatial_shapes=spatial_shapes
            )
            h_pyramids.append(h_b)

        # Batch-collate graph-enhanced features H1, H2, H3
        h_batched = [
            torch.cat([h_pyramids[b][lvl] for b in range(batch_size)], dim=0)
            for lvl in range(3)
        ]

        # In Stage 3, router is supervised, but features are visual-only for stability
        if self.current_stage == 3:
            cls_preds, reg_preds, cen_preds = self.det_head(list(features))
            return {
                "cls_preds": cls_preds,
                "reg_preds": reg_preds,
                "cen_preds": cen_preds,
                "features": features,
                "u_a": u_a_list,
                "u_e": u_e_list,
                "unc_preds": unc_preds,
                "graph_data": graph_data,
                "router_data": router_outputs,
                "stage": 3,
            }

        # 6. Stage 4: Uncertainty-Aware Feature Fusion (Zi = λi * Fi + (1 - λi) * Hi)
        z_features = self.fusion_gate(list(features), h_batched, u_a_list, u_e_list)

        # 7. Downstream FCOS Head on Fused Features Z
        cls_preds, reg_preds, cen_preds = self.det_head(z_features)

        return {
            "cls_preds": cls_preds,
            "reg_preds": reg_preds,
            "cen_preds": cen_preds,
            "features": features,
            "fused_features": z_features,
            "h_features": h_batched,
            "u_a": u_a_list,
            "u_e": u_e_list,
            "unc_preds": unc_preds,
            "graph_data": graph_data,
            "router_data": router_outputs,
            "stage": 4,
        }

