from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.losses.consistency_loss import ObjectCentricConsistencyLoss
from src.losses.detection_loss import FCOSDetectionLoss
from src.losses.graph_loss import GraphLaplacianLoss
from src.losses.router_loss import RouterSupervisionLoss
from src.losses.uncertainty_loss import UncertaintyLoss


class UCRMFGCombinedLoss(nn.Module):
    """Stage-aware objective: det + var + unc (1+), graph (2+), inst (3+), consistency (4).
    The benefit term is added by `add_benefit` once b_ij has been measured from the detection loss."""

    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        lambda_u: float = 0.5,
        lambda_var: float = 0.5,
        lambda_graph: float = 0.1,
        lambda_inst: float = 0.2,
        lambda_benefit: float = 0.2,
        lambda_consistency: float = 0.1,
        benefit_quantile: float = 0.75,
        router_pos_weight: float = 3.0,
    ):
        super().__init__()
        self.weights = {
            "loss_u": lambda_u,
            "loss_var": lambda_var,
            "loss_graph": lambda_graph,
            "loss_inst": lambda_inst,
            "loss_benefit": lambda_benefit,
            "loss_consistency": lambda_consistency,
        }
        self.det = FCOSDetectionLoss(strides)
        self.unc = UncertaintyLoss()
        self.graph = GraphLaplacianLoss()
        self.router = RouterSupervisionLoss(benefit_quantile, router_pos_weight)
        self.consistency = ObjectCentricConsistencyLoss(strides)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "UCRMFGCombinedLoss":
        keys = (
            "lambda_u", "lambda_var", "lambda_graph", "lambda_inst", "lambda_benefit",
            "lambda_consistency", "benefit_quantile", "router_pos_weight",
        )
        return cls(strides=cfg["model"]["strides"], **{k: cfg["loss"][k] for k in keys if k in cfg["loss"]})

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, List[torch.Tensor]],
        locations: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        head = outputs["head"]
        losses = self.det(
            head, targets["cls_targets"], targets["reg_targets"], targets["cen_targets"],
            targets["pos_masks"], locations,
        )
        losses.update(self.unc(
            outputs["unc_preds"], head["reg"], targets["cls_targets"], targets["reg_targets"], targets["pos_masks"]
        ))
        if outputs["graph"] is not None:
            losses["loss_graph"] = self.graph(outputs["graph"])
        if outputs["routing"] is not None and outputs["routing"].scores.requires_grad:
            losses["loss_inst"] = self.router.instance_loss(outputs["graph"], outputs["routing"], gt_boxes)
        if outputs["stage"] >= 4:
            losses["loss_consistency"] = self.consistency(head["cls"], gt_boxes)
        losses["loss_total"] = self._total(losses)
        return losses

    def add_benefit(
        self, losses: Dict[str, torch.Tensor], outputs: Dict[str, Any], benefit: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        losses["loss_benefit"] = self.router.benefit_loss(outputs["routing"], benefit)
        losses["loss_total"] = self._total(losses)
        return losses

    def _total(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = losses["loss_det"]
        for key, weight in self.weights.items():
            if key in losses:
                total = total + weight * losses[key]
        return total
