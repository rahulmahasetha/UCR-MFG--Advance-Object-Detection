from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterSupervisionLoss(nn.Module):
    """Dual Router Supervision: Weak Instance BCE (L_inst) + Top-Quantile Benefit Distillation (L_benefit)."""

    def __init__(self, benefit_quantile: float = 0.75, pos_weight_f3: float = 3.0):
        super().__init__()
        self.benefit_quantile = benefit_quantile
        self.pos_weight_f3 = pos_weight_f3

    def compute_instance_loss(
        self,
        s_ij: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        locations: List[torch.Tensor],
        gt_boxes: torch.Tensor,
        scale_splits: List[int]
    ) -> torch.Tensor:
        """Weak instance BCE for edges connecting nodes within the same GT object."""
        device = s_ij.device
        if gt_boxes.numel() == 0 or s_ij.numel() == 0:
            return torch.tensor(0.0, device=device)

        # Same-scale edges only (local and knn)
        samescale_mask = (edge_types == 0) | (edge_types == 1)
        if not samescale_mask.any():
            return torch.tensor(0.0, device=device)

        src = edge_index[0, samescale_mask]
        tgt = edge_index[1, samescale_mask]
        sub_s_ij = s_ij[samescale_mask]

        # Map node index to coordinate location
        all_locs = torch.cat(locations, dim=0)
        loc_src = all_locs[src]
        loc_tgt = all_locs[tgt]

        # Check if both inside same center-sampled GT box (r=0.5)
        # GT box centers:
        cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
        cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
        bw = (gt_boxes[:, 2] - gt_boxes[:, 0]) * 0.5
        bh = (gt_boxes[:, 3] - gt_boxes[:, 1]) * 0.5

        # Broadcast test: [E, M]
        src_in = (torch.abs(loc_src[:, 0:1] - cx.unsqueeze(0)) < bw.unsqueeze(0) / 2.0) & \
                 (torch.abs(loc_src[:, 1:2] - cy.unsqueeze(0)) < bh.unsqueeze(0) / 2.0)
        tgt_in = (torch.abs(loc_tgt[:, 0:1] - cx.unsqueeze(0)) < bw.unsqueeze(0) / 2.0) & \
                 (torch.abs(loc_tgt[:, 1:2] - cy.unsqueeze(0)) < bh.unsqueeze(0) / 2.0)

        # Both inside same box
        both_in = (src_in & tgt_in).any(dim=1).float()

        pos_weight = torch.tensor([self.pos_weight_f3], device=device)
        loss = F.binary_cross_entropy_with_logits(sub_s_ij, both_in, pos_weight=pos_weight)
        return loss

    def compute_benefit_loss(
        self,
        s_ij: torch.Tensor,
        benefit_grads: torch.Tensor
    ) -> torch.Tensor:
        """Top-quantile binary benefit distillation BCE."""
        device = s_ij.device
        if benefit_grads is None or benefit_grads.numel() == 0:
            return torch.tensor(0.0, device=device)

        b_ij = benefit_grads.detach()
        q_thresh = torch.quantile(b_ij, self.benefit_quantile)
        y_benefit = (b_ij >= q_thresh).float()

        loss = F.binary_cross_entropy_with_logits(s_ij, y_benefit)
        return loss

    def forward(
        self,
        router_outputs: List[Dict[str, Any]],
        graph_data: List[Dict[str, Any]],
        locations: List[torch.Tensor],
        gt_boxes_batch: List[torch.Tensor],
        benefit_grads_batch: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        device = router_outputs[0]["s_ij"].device
        inst_losses = []
        benefit_losses = []

        for b, (r_out, g_data) in enumerate(zip(router_outputs, graph_data)):
            s_ij = r_out["s_ij"]
            edge_index = r_out["edge_index"]
            edge_types = g_data["edge_types"]
            boxes_b = gt_boxes_batch[b].to(device)

            l_inst = self.compute_instance_loss(
                s_ij, edge_index, edge_types, locations, boxes_b, g_data["scale_splits"]
            )
            inst_losses.append(l_inst)

            b_grads = benefit_grads_batch[b] if benefit_grads_batch is not None else None
            l_ben = self.compute_benefit_loss(s_ij, b_grads)
            benefit_losses.append(l_ben)

        total_inst = sum(inst_losses) / len(inst_losses) if inst_losses else torch.tensor(0.0, device=device)
        total_ben = sum(benefit_losses) / len(benefit_losses) if benefit_losses else torch.tensor(0.0, device=device)

        return {
            "loss_inst": total_inst,
            "loss_benefit": total_ben,
        }
