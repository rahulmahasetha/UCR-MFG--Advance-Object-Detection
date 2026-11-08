from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn

from src.losses.detection_loss import FCOSDetectionLoss
from src.losses.uncertainty_loss import UncertaintyLoss
from src.losses.graph_loss import GraphLaplacianLoss
from src.losses.router_loss import RouterSupervisionLoss
from src.losses.consistency_loss import ObjectCentricConsistencyLoss


class UCRMFGCombinedLoss(nn.Module):
    """Unified Combined Loss Objective supporting all 4 curriculum stages."""

    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        lambda_u: float = 0.5,
        lambda_graph: float = 0.1,
        lambda_inst: float = 0.2,
        lambda_benefit: float = 0.2,
        lambda_consistency: float = 0.1
    ):
        super().__init__()
        self.strides = strides
        self.lambda_u = lambda_u
        self.lambda_g = lambda_graph
        self.lambda_inst = lambda_inst
        self.lambda_benefit = lambda_benefit
        self.lambda_c = lambda_consistency

        self.det_loss_fn = FCOSDetectionLoss(strides=strides)
        self.unc_loss_fn = UncertaintyLoss(strides=strides)
        self.graph_loss_fn = GraphLaplacianLoss()
        self.router_loss_fn = RouterSupervisionLoss()
        self.consistency_loss_fn = ObjectCentricConsistencyLoss(strides=strides)

    def forward(
        self,
        model_outputs: Dict[str, Any],
        targets: Dict[str, Any],
        locations: List[torch.Tensor],
        gt_boxes_batch: List[torch.Tensor],
        gt_labels_batch: List[torch.Tensor],
        benefit_grads_batch: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        stage = model_outputs.get("stage", 4)
        cls_preds = model_outputs["cls_preds"]
        reg_preds = model_outputs["reg_preds"]
        cen_preds = model_outputs["cen_preds"]
        device = cls_preds[0].device

        cls_targets = targets["cls_targets"]
        reg_targets = targets["reg_targets"]
        cen_targets = targets["cen_targets"]
        pos_masks = targets["pos_masks"]

        # 1. Detection loss (Always active)
        det_dict = self.det_loss_fn(
            cls_preds, reg_preds, cen_preds,
            cls_targets, reg_targets, cen_targets, pos_masks, locations
        )
        l_det = det_dict["loss_det"]

        # 2. Uncertainty loss (Stage 1+)
        unc_dict = self.unc_loss_fn(
            model_outputs["unc_preds"], reg_preds,
            cls_targets, reg_targets, pos_masks
        )
        l_u = unc_dict["loss_u"]

        total_loss = l_det + self.lambda_u * l_u
        loss_dict = {
            "loss_total": total_loss,
            "loss_det": l_det,
            "loss_cls": det_dict["loss_cls"],
            "loss_box": det_dict["loss_box"],
            "loss_cen": det_dict["loss_cen"],
            "loss_u": l_u,
            "loss_box_nll": unc_dict["loss_box_nll"],
            "loss_cls_nll": unc_dict["loss_cls_nll"],
            "loss_graph": torch.tensor(0.0, device=device),
            "loss_inst": torch.tensor(0.0, device=device),
            "loss_benefit": torch.tensor(0.0, device=device),
            "loss_consistency": torch.tensor(0.0, device=device),
        }

        # 3. Graph loss (Stage 2+)
        if stage >= 2 and model_outputs.get("graph_data") is not None:
            l_graph = self.graph_loss_fn(model_outputs["graph_data"])
            total_loss = total_loss + self.lambda_g * l_graph
            loss_dict["loss_graph"] = l_graph

        # 4. Router losses (Stage 3+)
        if stage >= 3 and model_outputs.get("router_data") is not None:
            r_dict = self.router_loss_fn(
                model_outputs["router_data"],
                model_outputs["graph_data"],
                locations,
                gt_boxes_batch,
                benefit_grads_batch
            )
            l_inst = r_dict["loss_inst"]
            l_ben = r_dict["loss_benefit"]
            total_loss = total_loss + self.lambda_inst * l_inst + self.lambda_benefit * l_ben
            loss_dict["loss_inst"] = l_inst
            loss_dict["loss_benefit"] = l_ben

        # 5. Object-centric consistency loss (Stage 4)
        if stage >= 4:
            l_c = self.consistency_loss_fn(cls_preds, gt_boxes_batch)
            total_loss = total_loss + self.lambda_c * l_c
            loss_dict["loss_consistency"] = l_c

        loss_dict["loss_total"] = total_loss
        return loss_dict

