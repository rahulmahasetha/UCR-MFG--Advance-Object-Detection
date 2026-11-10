import pytest
import torch

from src.models.fusion import UncertaintyFusionGate
from src.models.message_passing import BenefitAwareMPNN


def toy_inputs():
    torch.manual_seed(0)
    nodes = torch.randn(6, 64)
    edge_index = torch.tensor([[0, 0, 0, 1, 2], [1, 2, 3, 0, 4]])
    scores = torch.randn(5, requires_grad=True)
    prob = torch.sigmoid(scores)
    selected = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    mask = selected + prob - prob.detach()
    return nodes, edge_index, mask, scores, torch.rand(6), torch.rand(6)


def test_mpnn_shapes_and_straight_through_gradient():
    mpnn = BenefitAwareMPNN(64, 256)
    nodes, edge_index, mask, scores, ua, ue = toy_inputs()
    out = mpnn(nodes, edge_index, mask, scores, ua, ue)
    assert out.shape == (6, 256)
    out.sum().backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0


def test_unselected_edge_receives_benefit_gradient():
    mpnn = BenefitAwareMPNN(64, 256)
    nodes, edge_index, mask, scores, ua, ue = toy_inputs()
    out = mpnn(nodes, edge_index, mask, scores, ua, ue)
    (grad,) = torch.autograd.grad(out.pow(2).sum(), mask)
    assert grad[2].abs() > 0, "edge with mask 0 must still get dL/dm for benefit labels"


def test_unselected_edge_has_no_forward_effect():
    mpnn = BenefitAwareMPNN(64, 256).eval()
    nodes, edge_index, mask, scores, ua, ue = toy_inputs()
    dense = mpnn(nodes, edge_index, mask.detach(), scores.detach(), ua, ue)
    keep = mask.detach() > 0.5
    sparse = mpnn(nodes, edge_index[:, keep], mask.detach()[keep], scores.detach()[keep], ua, ue)
    assert torch.allclose(dense, sparse, atol=1e-5)


@pytest.mark.parametrize("mode", ["gate", "fixed", "add"])
def test_fusion_modes(mode):
    fusion = UncertaintyFusionGate(256, mode=mode)
    f, h = torch.randn(2, 256, 8, 8), torch.randn(2, 256, 8, 8)
    (z,) = fusion([f], [h], [torch.rand(2, 1, 8, 8)], [torch.rand(2, 1, 8, 8)])
    assert z.shape == f.shape


def test_fusion_gate_starts_visual():
    fusion = UncertaintyFusionGate(256)
    f, h = torch.randn(1, 256, 4, 4), torch.zeros(1, 256, 4, 4)
    (z,) = fusion([f], [h], [torch.zeros(1, 1, 4, 4)], [torch.zeros(1, 1, 4, 4)])
    ratio = (z / f).median()
    assert 0.7 < ratio < 0.99
