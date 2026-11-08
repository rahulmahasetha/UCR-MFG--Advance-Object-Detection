import torch
import pytest
from src.models.message_passing import BenefitAwareMPNN
from src.models.fusion import UncertaintyFusionGate


def test_benefit_aware_mpnn_and_gradient():
    mpnn = BenefitAwareMPNN(graph_dim=64, attn_dim=32, out_channels=256, k_total=4)

    num_nodes = 21  # 16 + 4 + 1
    nodes = torch.randn(num_nodes, 64)
    src = torch.tensor([0, 1, 2, 3])
    tgt = torch.tensor([1, 2, 3, 0])
    edge_index = torch.stack([src, tgt], dim=0)

    s_ij = torch.randn(4, requires_grad=True)
    m_st = torch.ones(4) + (torch.sigmoid(s_ij) - torch.sigmoid(s_ij).detach())
    ue_flat = torch.rand(num_nodes)

    scale_splits = [16, 4, 1]
    spatial_shapes = [(4, 4), (2, 2), (1, 1)]

    h_pyramid = mpnn(nodes, edge_index, m_st, s_ij, ue_flat, scale_splits, spatial_shapes)
    assert len(h_pyramid) == 3
    assert h_pyramid[0].shape == (1, 256, 4, 4)
    assert h_pyramid[1].shape == (1, 256, 2, 2)
    assert h_pyramid[2].shape == (1, 256, 1, 1)

    loss = h_pyramid[0].sum()
    loss.backward()
    assert s_ij.grad is not None
    assert s_ij.grad.abs().sum().item() > 0.0, "Surrogate gradient did not flow through MPNN into s_ij!"


def test_fusion_gate_bounds():
    fusion = UncertaintyFusionGate(channels=256)
    f1 = torch.randn(2, 256, 8, 8)
    h1 = torch.randn(2, 256, 8, 8)
    ua1 = torch.rand(2, 1, 8, 8)
    ue1 = torch.rand(2, 1, 8, 8)

    z_list = fusion([f1], [h1], [ua1], [ue1])
    assert len(z_list) == 1
    assert z_list[0].shape == (2, 256, 8, 8)

