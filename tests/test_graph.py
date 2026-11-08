import torch
import pytest
from src.models.graph_builder import MultiScaleGraphBuilder


def test_graph_builder_construction():
    builder = MultiScaleGraphBuilder(in_channels=256, graph_dim=64, r_local=2, k_feat=4, f1_block_size=8)
    f1 = torch.randn(1, 256, 8, 8)
    f2 = torch.randn(1, 256, 4, 4)
    f3 = torch.randn(1, 256, 2, 2)

    graphs = builder((f1, f2, f3))
    assert len(graphs) == 1
    g0 = graphs[0]

    # Total nodes = 8*8 + 4*4 + 2*2 = 64 + 16 + 4 = 84
    assert g0["nodes"].shape == (84, 64)
    assert g0["edge_index"].shape[0] == 2
    assert g0["edge_index"].shape[1] > 0
    assert g0["kij"].shape[0] == g0["edge_index"].shape[1]
    assert g0["wij"].shape[0] == g0["edge_index"].shape[1]

    # Weights in [0, 1]
    assert g0["kij"].min() >= 0.0 and g0["kij"].max() <= 1.0
    assert g0["wij"].min() >= 0.0 and g0["wij"].max() <= 1.0

