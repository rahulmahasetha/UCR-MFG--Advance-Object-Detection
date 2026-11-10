from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.graph_ops import KNN, LOCAL, Graph, Routing


@torch.no_grad()
def node_instance_ids(graph: Graph, gt_boxes: List[torch.Tensor], center_ratio: float = 0.5) -> torch.Tensor:
    """Smallest GT box whose central region contains each node, or -1. Ids are unique across the batch."""
    n = graph.num_nodes
    ids = torch.full((graph.batch_size * n,), -1, dtype=torch.long, device=graph.positions.device)
    offset = 0
    for b, boxes in enumerate(gt_boxes):
        if boxes.numel() == 0:
            continue
        boxes = boxes.to(graph.positions.device)
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        half = (boxes[:, 2:] - boxes[:, :2]) * (center_ratio / 2)
        inside = ((graph.positions[:, None] - center[None]).abs() < half[None]).all(-1)
        area = (boxes[:, 2:] - boxes[:, :2]).prod(-1).expand(n, -1).masked_fill(~inside, torch.inf)
        min_area, box_idx = area.min(dim=1)
        ids[b * n:(b + 1) * n] = torch.where(torch.isfinite(min_area), box_idx + offset, -1)
        offset += boxes.shape[0]
    return ids


@torch.no_grad()
def same_instance_edges(graph: Graph, gt_boxes: List[torch.Tensor]) -> torch.Tensor:
    ids = node_instance_ids(graph, gt_boxes)
    recv, send = graph.edge_index
    return (ids[recv] >= 0) & (ids[recv] == ids[send])


class RouterSupervisionLoss(nn.Module):
    """Router supervision.

    L_inst: weak BCE that same-scale edges inside one object are relevant.
    L_benefit: BCE towards edges whose first-order effect b_ij = -dL_det/dm_ij lowers the detection loss
    (top quantile among positive-benefit candidates).
    """

    def __init__(self, benefit_quantile: float = 0.75, pos_weight: float = 3.0):
        super().__init__()
        self.benefit_quantile = benefit_quantile
        self.register_buffer("pos_weight", torch.tensor([pos_weight]), persistent=False)

    def instance_loss(self, graph: Graph, routing: Routing, gt_boxes: List[torch.Tensor]) -> torch.Tensor:
        same_scale = (graph.edge_types == LOCAL) | (graph.edge_types == KNN)
        if not same_scale.any():
            return routing.scores.sum() * 0.0
        target = same_instance_edges(graph, gt_boxes)[same_scale].float()
        return F.binary_cross_entropy_with_logits(
            routing.scores[same_scale], target, pos_weight=self.pos_weight
        )

    def benefit_loss(self, routing: Routing, benefit: Optional[torch.Tensor]) -> torch.Tensor:
        if benefit is None or benefit.numel() == 0:
            return routing.scores.sum() * 0.0
        return F.binary_cross_entropy_with_logits(routing.scores, benefit_targets(benefit, self.benefit_quantile))


@torch.no_grad()
def benefit_targets(benefit: torch.Tensor, quantile: float) -> torch.Tensor:
    benefit = benefit.float()
    positive = benefit > 0
    if not positive.any():
        return torch.zeros_like(benefit)
    values = benefit[positive].sort().values
    threshold = values[min(int(quantile * values.numel()), values.numel() - 1)]
    return (positive & (benefit >= threshold)).float()
