from typing import Sequence

import torch
import torch.nn as nn

from src.models.graph_ops import NUM_EDGE_TYPES, Graph, Routing


class UCRouter(nn.Module):
    """C2: Uncertainty-conditioned relational routing.

    Each candidate edge is scored from a 140-d descriptor [h_i, h_j, k_ij, Ua_i, Ue_i, Ua_j, Ue_j, geometry, type].
    The first layer is factorised over receiver, sender and edge terms, which is exact and avoids
    materialising the descriptor. A per-receiver, per-type top-K keeps the best edges; the budget grows
    with receiver uncertainty so ambiguous locations gather more context. Straight-through masks keep
    the selection differentiable.
    """

    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 64,
        budgets: Sequence[int] = (4, 2, 2, 2),
        k_gain: float = 2.0,
        use_uncertainty: bool = True,
        enabled: bool = True,
        type_dim: int = 4,
    ):
        super().__init__()
        self.register_buffer("budgets", torch.tensor(budgets, dtype=torch.long), persistent=False)
        self.k_gain = k_gain
        self.use_uncertainty = use_uncertainty
        self.enabled = enabled
        self.type_emb = nn.Embedding(NUM_EDGE_TYPES, type_dim)
        self.recv_proj = nn.Linear(node_dim, hidden_dim)
        self.send_proj = nn.Linear(node_dim, hidden_dim, bias=False)
        self.edge_proj = nn.Linear(1 + 4 + 3 + type_dim, hidden_dim, bias=False)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, graph: Graph, ua: torch.Tensor, ue: torch.Tensor) -> Routing:
        recv, send = graph.edge_index
        if not self.enabled:
            ones = torch.ones(recv.numel(), device=recv.device)
            return Routing(scores=torch.zeros_like(ones), mask=ones, selected=ones.bool())
        if not self.use_uncertainty:
            ua, ue = torch.zeros_like(ua), torch.zeros_like(ue)

        edge_feats = torch.cat(
            [
                graph.kij.unsqueeze(1),
                torch.stack([ua[recv], ue[recv], ua[send], ue[send]], dim=1),
                graph.edge_geom,
                self.type_emb(graph.edge_types),
            ],
            dim=1,
        ).to(graph.nodes.dtype)
        hidden = self.recv_proj(graph.nodes)[recv] + self.send_proj(graph.nodes)[send] + self.edge_proj(edge_feats)
        scores = self.out(hidden).squeeze(-1).float()

        budget = self.budgets[graph.edge_types]
        if self.use_uncertainty and self.k_gain > 0:
            budget = budget + torch.round(self.k_gain * torch.maximum(ua, ue)[recv]).long()
        selected = topk_per_group(scores.detach(), recv * NUM_EDGE_TYPES + graph.edge_types, budget)

        prob = torch.sigmoid(scores)
        mask = selected.float() + prob - prob.detach()
        return Routing(scores=scores, mask=mask, selected=selected)


@torch.no_grad()
def topk_per_group(scores: torch.Tensor, group: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    """Boolean mask of the `budget[e]` highest-scoring edges within each group."""
    num = scores.numel()
    if num == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    score_rank = torch.empty(num, dtype=torch.long, device=scores.device)
    score_rank[torch.argsort(scores, descending=True)] = torch.arange(num, device=scores.device)
    order = torch.argsort(group * num + score_rank)

    sorted_group = group[order]
    counts = torch.bincount(sorted_group)
    starts = counts.cumsum(0) - counts
    rank = torch.arange(num, device=scores.device) - starts[sorted_group]

    selected = torch.zeros(num, dtype=torch.bool, device=scores.device)
    selected[order] = rank < budget[order]
    return selected
