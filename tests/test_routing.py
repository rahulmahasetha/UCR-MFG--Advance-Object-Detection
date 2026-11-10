import torch

from src.models.graph_builder import MultiScaleGraphBuilder
from src.models.graph_ops import NUM_EDGE_TYPES
from src.models.router import UCRouter, topk_per_group


def brute_force_topk(scores, group, budget):
    out = torch.zeros_like(scores, dtype=torch.bool)
    for g in group.unique():
        idx = torch.nonzero(group == g).squeeze(1)
        k = int(budget[idx[0]])
        out[idx[scores[idx].argsort(descending=True)[:k]]] = True
    return out


def test_vectorised_topk_matches_brute_force():
    torch.manual_seed(0)
    scores = torch.randn(500)
    group = torch.randint(0, 40, (500,))
    budget = torch.randint(0, 6, (40,))[group]
    assert torch.equal(topk_per_group(scores, group, budget), brute_force_topk(scores, group, budget))


def make_graph():
    builder = MultiScaleGraphBuilder(256, 64, (8, 16, 32), r_local=1, k_feat=4, block_size=4)
    return builder([torch.randn(1, 256, 8, 8), torch.randn(1, 256, 4, 4), torch.randn(1, 256, 2, 2)])


def test_router_budget_and_straight_through_gradient():
    graph = make_graph()
    n = graph.nodes.shape[0]
    router = UCRouter(64, budgets=(2, 1, 1, 1), k_gain=0.0)
    out = router(graph, torch.rand(n), torch.rand(n))

    group = graph.edge_index[0] * NUM_EDGE_TYPES + graph.edge_types
    per_group = torch.bincount(group[out.selected], minlength=int(group.max()) + 1)
    assert per_group.max() <= 2
    assert torch.allclose(out.mask.detach(), out.selected.float())

    out.mask.sum().backward()
    assert sum(p.grad.abs().sum() for p in router.parameters() if p.grad is not None) > 0


def test_uncertain_receivers_get_larger_budgets():
    graph = make_graph()
    n = graph.nodes.shape[0]
    router = UCRouter(64, budgets=(1, 1, 1, 1), k_gain=3.0)
    calm = router(graph, torch.zeros(n), torch.zeros(n)).selected.sum()
    uncertain = router(graph, torch.ones(n), torch.ones(n)).selected.sum()
    assert uncertain > calm


def test_disabled_router_keeps_all_edges():
    graph = make_graph()
    n = graph.nodes.shape[0]
    out = UCRouter(64, enabled=False)(graph, torch.rand(n), torch.rand(n))
    assert out.selected.all() and torch.all(out.mask == 1)
