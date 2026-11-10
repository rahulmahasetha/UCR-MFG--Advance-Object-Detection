import torch

from src.models.graph_builder import MultiScaleGraphBuilder
from src.models.graph_ops import BOTTOM_UP, KNN, LOCAL, TOP_DOWN, flatten_levels, to_pyramid


def make_features(batch=2):
    return [torch.randn(batch, 256, 8, 8), torch.randn(batch, 256, 4, 4), torch.randn(batch, 256, 2, 2)]


def test_graph_structure_and_weights():
    builder = MultiScaleGraphBuilder(256, 64, (8, 16, 32), r_local=1, k_feat=4, block_size=4)
    graph = builder(make_features())
    n = 64 + 16 + 4

    assert graph.nodes.shape == (2 * n, 64)
    assert graph.num_nodes == n
    assert graph.edge_index.max() < 2 * n
    assert graph.edge_geom.shape == (graph.edge_index.shape[1], 3)
    assert 0 <= graph.kij.min() and graph.kij.max() <= 1
    assert set(graph.edge_types.unique().tolist()) == {LOCAL, KNN, TOP_DOWN, BOTTOM_UP}

    recv, send = graph.edge_index
    assert ((recv // n) == (send // n)).all(), "edges must stay inside one image"
    keys = recv * 2 * n + send
    assert keys.unique().numel() == keys.numel(), "candidate edges must be unique"

    knn = graph.edge_types == KNN
    assert (graph.edge_geom[knn, 2] > 0).all(), "kNN edges carry geometry"


def test_every_fine_node_has_a_parent():
    builder = MultiScaleGraphBuilder(256, 64, (8, 16, 32), r_local=1, k_feat=2, block_size=4)
    graph = builder(make_features(batch=1))
    top_down = graph.edge_types == TOP_DOWN
    receivers = graph.edge_index[0, top_down].unique()
    assert receivers.numel() == 64 + 16
    parents = graph.edge_index[0, graph.edge_types == BOTTOM_UP]
    assert torch.bincount(parents).max() == 4


def test_pyramid_round_trip():
    feats = make_features()
    flat = flatten_levels(feats)
    back = to_pyramid(flat, 2, [tuple(f.shape[-2:]) for f in feats])
    assert all(torch.equal(a, b) for a, b in zip(feats, back))
