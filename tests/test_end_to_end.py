import torch
import pytest

from src.models.ucr_mfg import UCRMFG
from src.losses.total_loss import UCRMFGCombinedLoss


def test_ucr_mfg_full_pipeline_all_stages():
    model = UCRMFG(num_classes=10, fpn_channels=256, graph_dim=64, mc_passes=2, pretrained=False)
    loss_fn = UCRMFGCombinedLoss()

    img = torch.randn(2, 3, 128, 128)
    boxes = [torch.tensor([[10.0, 10.0, 30.0, 30.0]]), torch.empty((0, 4))]
    labels = [torch.tensor([3]), torch.empty((0,), dtype=torch.long)]
    ignore_masks = torch.zeros((2, 128, 128), dtype=torch.uint8)

    # Grid locations for shapes [(16, 16), (8, 8), (4, 4)]
    shapes = [(16, 16), (8, 8), (4, 4)]
    locations = model.det_head.get_grid_locations(shapes, img.device)
    cls_t, reg_t, cen_t, pos_m = model.det_head.assign_targets(locations, boxes, labels, ignore_masks)
    targets = {
        "cls_targets": cls_t,
        "reg_targets": reg_t,
        "cen_targets": cen_t,
        "pos_masks": pos_m,
    }

    # Test all 4 curriculum stages
    for stage in [1, 2, 3, 4]:
        model.set_curriculum_stage(stage)
        outputs = model(img)
        assert outputs["stage"] == stage

        loss_dict = loss_fn(outputs, targets, locations, boxes, labels)
        assert not torch.isnan(loss_dict["loss_total"])
        assert loss_dict["loss_total"].item() > 0.0

        # Backpropagation check
        model.zero_grad()
        loss_dict["loss_total"].backward()

        # Check gradients
        backbone_grad = any([p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.backbone.parameters()])
        assert backbone_grad is True, f"Stage {stage}: Backbone received zero gradient!"

