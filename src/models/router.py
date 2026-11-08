from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class UCRouter(nn.Module):
    """Contribution 2: Uncertainty-Conditioned Relational Router.
    
    Selects informative subgraphs via straight-through Top-K routing conditioned on 
    node features, spatial geometry, and dual predictive uncertainties (Ua, Ue).
    """

    def __init__(
        self,
        input_dim: int = 140,
        hidden_dim: int = 64,
        k_local: int = 4,
        k_feat: int = 2,
        k_cross: int = 2,
        num_edge_types: int = 4,
        edge_type_dim: int = 4
    ):
        super().__init__()
        self.k_budgets = {
            0: k_local,   # Local window
            1: k_feat,    # Feature kNN
            2: k_cross,   # Cross-scale
            3: k_cross,   # Cross-scale
        }
        self.edge_type_emb = nn.Embedding(num_edge_types, edge_type_dim)

        self.router_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def assemble_input_vector(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        kij: torch.Tensor,
        deltas: torch.Tensor,
        edge_types: torch.Tensor,
        ua: torch.Tensor,
        ue: torch.Tensor
    ) -> torch.Tensor:
        """Assembles the exact 140-dimensional candidate edge input vector."""
        src, tgt = edge_index[0], edge_index[1]

        h_i = nodes[src]                                  # [E, 64]
        h_j = nodes[tgt]                                  # [E, 64]
        k_val = kij.unsqueeze(1)                          # [E, 1]
        ua_i = ua[src].unsqueeze(1)                       # [E, 1]
        ue_i = ue[src].unsqueeze(1)                       # [E, 1]
        ua_j = ua[tgt].unsqueeze(1)                       # [E, 1]
        ue_j = ue[tgt].unsqueeze(1)                       # [E, 1]
        geom = deltas                                     # [E, 3]
        e_emb = self.edge_type_emb(edge_types)            # [E, 4]

        x_140 = torch.cat([h_i, h_j, k_val, ua_i, ue_i, ua_j, ue_j, geom, e_emb], dim=-1)
        return x_140

    def straight_through_topk(
        self,
        s_ij: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs deterministic per-type Top-K selection and preserves ST surrogate gradients."""
        device = s_ij.device
        src = edge_index[0]
        m_hard = torch.zeros_like(s_ij)

        # Per edge type Top-K selection
        for e_type, k_val in self.k_budgets.items():
            type_mask = (edge_types == e_type)
            if not type_mask.any() or k_val <= 0:
                continue

            type_indices = torch.nonzero(type_mask).squeeze(1)
            type_src = src[type_indices]
            type_scores = s_ij[type_indices]

            # Unique source nodes for this type
            for u_src in torch.unique(type_src):
                node_edge_mask = (type_src == u_src)
                candidate_sub_indices = type_indices[node_edge_mask]
                sub_scores = type_scores[node_edge_mask]

                k = min(k_val, sub_scores.size(0))
                if k > 0:
                    _, top_sub_idx = torch.topk(sub_scores, k=k, largest=True)
                    m_hard[candidate_sub_indices[top_sub_idx]] = 1.0

        # Straight-Through Estimator:
        # Forward pass is binary m_hard; backward pass flows through sigmoid(s_ij)
        prob_s = torch.sigmoid(s_ij)
        m_st = m_hard + (prob_s - prob_s.detach())
        return m_st, m_hard

    def forward(
        self,
        graph_dict: Dict[str, Any],
        ua_flat: torch.Tensor,
        ue_flat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Routes candidate graph edges and produces Straight-Through mask tensor."""
        nodes = graph_dict["nodes"]
        edge_index = graph_dict["edge_index"]
        kij = graph_dict["kij"]
        deltas = graph_dict["edge_deltas"]
        edge_types = graph_dict["edge_types"]

        x_140 = self.assemble_input_vector(nodes, edge_index, kij, deltas, edge_types, ua_flat, ue_flat)
        s_ij = self.router_mlp(x_140).squeeze(-1)  # [E]

        m_st, m_hard = self.straight_through_topk(s_ij, edge_index, edge_types, nodes.shape[0])

        return {
            "s_ij": s_ij,
            "m_st": m_st,
            "m_hard": m_hard,
            "edge_index": edge_index,
        }

