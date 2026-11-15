from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.models.backbone import ResNetFPNBackbone
from src.models.detection_head import FCOSHead
from src.models.fusion import UncertaintyFusionGate
from src.models.graph_builder import MultiScaleGraphBuilder
from src.models.graph_ops import flatten_levels, to_pyramid
from src.models.message_passing import BenefitAwareMPNN
from src.models.router import UCRouter
from src.models.uncertainty_head import PreGraphUncertaintyHead


class UCRMFG(nn.Module):
    """UCR-MFG: Uncertainty-Conditioned Relational Routing for Multi-Scale Fuzzy Graph Object Detection.

    Curriculum stages: 1 detector + C1, 2 adds the fuzzy graph, 3 adds C2 + C3 with a small fixed
    residual of graph features, 4 enables the C4 fusion gate.
    """

    def __init__(
        self,
        num_classes: int = 10,
        fpn_channels: int = 256,
        graph_dim: int = 64,
        strides: Sequence[int] = (8, 16, 32),
        pretrained: bool = False,
        freeze_bn: bool = True,
        frozen_stages: int = 1,
        uncertainty: Optional[Dict[str, Any]] = None,
        graph: Optional[Dict[str, Any]] = None,
        router: Optional[Dict[str, Any]] = None,
        mpnn: Optional[Dict[str, Any]] = None,
        fusion_mode: str = "gate",
        stage3_residual: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = list(strides)
        self.stage = 4
        self.stage3_residual = stage3_residual

        self.backbone = ResNetFPNBackbone(pretrained, fpn_channels, freeze_bn=freeze_bn, frozen_stages=frozen_stages)
        self.unc_head = PreGraphUncertaintyHead(fpn_channels, num_classes, **(uncertainty or {}))
        self.graph_builder = MultiScaleGraphBuilder(fpn_channels, graph_dim, self.strides, **(graph or {}))
        self.router = UCRouter(graph_dim, **(router or {}))
        self.mpnn = BenefitAwareMPNN(graph_dim, fpn_channels, **(mpnn or {}))
        self.fusion_gate = UncertaintyFusionGate(fpn_channels, mode=fusion_mode)
        self.det_head = FCOSHead(fpn_channels, num_classes, self.strides)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], num_classes: int) -> "UCRMFG":
        m = cfg["model"]
        r, g = m.get("router", {}), m.get("graph", {})
        return cls(
            num_classes=num_classes,
            fpn_channels=m["fpn_channels"],
            graph_dim=m["graph_dim"],
            strides=m["strides"],
            pretrained=m.get("pretrained", False),
            freeze_bn=m.get("freeze_bn", True),
            frozen_stages=m.get("frozen_stages", 1),
            uncertainty=m.get("uncertainty"),
            graph={k: g[k] for k in ("r_local", "k_feat", "alpha", "block_size") if k in g},
            router={
                "hidden_dim": r.get("hidden_dim", 64),
                "budgets": (r.get("k_local", 4), r.get("k_feat", 2), r.get("k_cross", 2), r.get("k_cross", 2)),
                "k_gain": r.get("k_gain", 2.0),
                "use_uncertainty": r.get("use_uncertainty", True),
                "enabled": r.get("enabled", True),
            },
            mpnn={"use_gate": m.get("mpnn", {}).get("benefit_gate", True)},
            fusion_mode=m.get("fusion", {}).get("mode", "gate"),
            stage3_residual=m.get("fusion", {}).get("stage3_residual", 0.1),
        )

    @torch.no_grad()
    def load_coco_fcos(self, coco_class_ids: Optional[Sequence[int]] = None) -> Dict[str, int]:
        """Initialise backbone, FPN (P3-P5) and FCOS head towers from torchvision's COCO-trained FCOS-R50-FPN.

        coco_class_ids[i] is the COCO category index whose classifier initialises our class i (e.g. UAVDT
        car/truck/bus -> 3/8/6). The box-regression output, centerness and graph/uncertainty modules stay fresh
        because their parameterisation differs from torchvision's.
        """
        from torchvision.models.detection import FCOS_ResNet50_FPN_Weights

        src = FCOS_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=True)
        mapped = {}
        for k, v in src.items():
            if k.startswith("backbone.body."):
                mapped["backbone." + k[len("backbone.body."):]] = v
        for i, (lat, smooth) in enumerate((("lat_c3", "smooth_p3"), ("lat_c4", "smooth_p4"), ("lat_c5", "smooth_p5"))):
            for p in ("weight", "bias"):
                mapped[f"backbone.{lat}.{p}"] = src[f"backbone.fpn.inner_blocks.{i}.{p}"]
                mapped[f"backbone.{smooth}.{p}"] = src[f"backbone.fpn.layer_blocks.{i}.{p}"]
        for ours, theirs in (("cls_tower", "classification_head.conv"), ("reg_tower", "regression_head.conv")):
            for k, v in src.items():
                if k.startswith(f"head.{theirs}."):
                    mapped[f"det_head.{ours}." + k[len(f"head.{theirs}."):]] = v
        if coco_class_ids is not None:
            if len(coco_class_ids) != self.num_classes:
                raise ValueError(f"coco_class_ids has {len(coco_class_ids)} entries for {self.num_classes} classes")
            idx = torch.as_tensor(list(coco_class_ids))
            mapped["det_head.cls_pred.weight"] = src["head.classification_head.cls_logits.weight"][idx]
            mapped["det_head.cls_pred.bias"] = src["head.classification_head.cls_logits.bias"][idx]

        own = self.state_dict()
        mismatched = [k for k, v in mapped.items() if k not in own or own[k].shape != v.shape]
        if mismatched:
            raise RuntimeError(f"COCO FCOS weights do not fit: {mismatched[:5]}")
        self.load_state_dict(mapped, strict=False)
        return {"loaded_tensors": len(mapped)}

    def set_curriculum_stage(self, stage: int) -> None:
        if stage not in (1, 2, 3, 4):
            raise ValueError(f"stage must be 1-4, got {stage}")
        self.stage = stage

    def forward(
        self,
        images: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        dense_routing: bool = False,
    ) -> Dict[str, Any]:
        stage = self.stage
        features = list(self.backbone(images))
        u_a, u_e, unc_preds = self.unc_head(features, generator, compute_epistemic=stage >= 2 or not self.training)
        out = {"stage": stage, "u_a": u_a, "u_e": u_e, "unc_preds": unc_preds, "graph": None, "routing": None}

        fused = features
        if stage >= 3 or (stage == 2 and self.training):
            out["graph"] = graph = self.graph_builder(features)
        if stage >= 3:
            ua, ue = flatten_levels(u_a).squeeze(1), flatten_levels(u_e).squeeze(1)
            routing = self.router(graph, ua, ue)
            active = slice(None) if dense_routing else routing.selected
            h = self.mpnn(
                graph.nodes, graph.edge_index[:, active], routing.mask[active], routing.scores[active], ua, ue
            )
            h = [x.to(f.dtype) for x, f in zip(to_pyramid(h, graph.batch_size, graph.shapes), features)]
            if stage == 3:
                fused = [f + self.stage3_residual * x for f, x in zip(features, h)]
            else:
                fused = self.fusion_gate(features, h, u_a, u_e)
            out["routing"] = routing

        out["head"] = self.det_head(fused)
        return out
