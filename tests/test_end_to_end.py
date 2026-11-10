import pytest
import torch

from src.engine import LocationCache, assign_targets, measure_benefit
from src.losses.total_loss import UCRMFGCombinedLoss
from src.models.postprocess import postprocess
from src.models.ucr_mfg import UCRMFG


def make_batch():
    return {
        "images": torch.randn(2, 3, 128, 128),
        "boxes": [torch.tensor([[10.0, 10.0, 40.0, 40.0], [60.0, 60.0, 120.0, 110.0]]), torch.empty((0, 4))],
        "labels": [torch.tensor([3, 5]), torch.empty((0,), dtype=torch.long)],
        "ignore_masks": torch.zeros((2, 128, 128), dtype=torch.uint8),
    }


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return UCRMFG(num_classes=10, uncertainty={"mc_passes": 2}, graph={"r_local": 1, "k_feat": 4, "block_size": 8})


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_training_step_all_stages(model, stage):
    loss_fn = UCRMFGCombinedLoss()
    locations = LocationCache(model.det_head)
    batch = make_batch()
    model.train()
    model.set_curriculum_stage(stage)

    dense = stage >= 3
    outputs = model(batch["images"], dense_routing=dense)
    locs = locations(outputs["head"]["cls"])
    losses = loss_fn(outputs, assign_targets(model, locs, batch, "cpu"), locs, batch["boxes"])
    if dense:
        benefit = measure_benefit(losses["loss_det"], outputs["routing"])
        assert benefit.shape == outputs["routing"].scores.shape
        assert benefit.abs().sum() > 0, "detection loss must reach the routing mask"
        losses = loss_fn.add_benefit(losses, outputs, benefit)

    assert torch.isfinite(losses["loss_total"])
    model.zero_grad()
    losses["loss_total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    if stage >= 3:
        assert model.router.recv_proj.weight.grad.abs().sum() > 0
        assert model.mpnn.value[0].weight.grad.abs().sum() > 0
    if stage == 4:
        assert model.fusion_gate.gate_conv[0].weight.grad.abs().sum() > 0


def test_inference_returns_detections_with_uncertainty(model):
    model.eval()
    model.set_curriculum_stage(4)
    images = make_batch()["images"]
    with torch.no_grad():
        outputs = model(images)
    locs = LocationCache(model.det_head)(outputs["head"]["cls"])
    dets = postprocess(outputs, locs, model.strides, images.shape[-2:], score_thresh=0.0, max_dets=50)
    assert len(dets) == 2
    for det in dets:
        n = det["scores"].numel()
        assert 0 < n <= 50
        assert det["box_std"].shape == (n, 4) and det["u_a"].shape == det["u_e"].shape == (n,)
        assert (det["boxes"][:, 2:] >= det["boxes"][:, :2]).all()
