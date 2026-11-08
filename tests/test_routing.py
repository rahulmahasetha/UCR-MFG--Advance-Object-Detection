import torch
import pytest
from src.models.router import UCRouter


def test_straight_through_router_gradient():
    router = UCRouter(input_dim=140, hidden_dim=64, k_local=2, k_feat=2, k_cross=2)

    num_nodes = 20
    nodes = torch.randn(num_nodes, 64)
    # Create simple cyclic edges
    src = torch.arange(num_nodes)
    tgt = (src + 1) % num_nodes
    edge_index = torch.stack([src, tgt], dim=0)

    kij = torch.ones(num_nodes) * 0.5
    deltas = torch.zeros(num_nodes, 3)
    edge_types = torch.zeros(num_nodes, dtype=torch.long)

    graph_dict = {
        "nodes": nodes,
        "edge_index": edge_index,
        "kij": kij,
        "edge_deltas": deltas,
        "edge_types": edge_types,
    }

    ua_flat = torch.rand(num_nodes)
    ue_flat = torch.rand(num_nodes)

    out = router(graph_dict, ua_flat, ue_flat)
    s_ij = out["s_ij"]
    m_st = out["m_st"]
    m_hard = out["m_hard"]

    # 1. Forward pass hard values are binary
    assert torch.all((m_hard == 0.0) | (m_hard == 1.0))

    # 2. Straight-Through gradient check
    loss = (m_st * 2.0).sum()
    loss.backward()

    # The router parameters must have received non-zero gradients!
    router_grad_norm = sum([p.grad.abs().sum().item() for p in router.parameters() if p.grad is not None])
    assert router_grad_norm > 0.0, "Straight-Through gradient failed: Router received zero gradient!"

