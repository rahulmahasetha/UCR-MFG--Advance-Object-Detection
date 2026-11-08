from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BenefitAwareMPNN(nn.Module):
    """Contribution 3: Benefit-Aware Multi-Scale Message Passing.
    
    Aggregates messages in 64-d projected space using straight-through masked attention
    and dynamic benefit gating, then linearly expands node representations back to C=256.
    """

    def __init__(
        self,
        graph_dim: int = 64,
        attn_dim: int = 32,
        out_channels: int = 256,
        k_total: int = 8
    ):
        super().__init__()
        self.graph_dim = graph_dim
        self.attn_dim = attn_dim
        self.out_channels = out_channels
        self.k_total = k_total

        # Message transformation MLP
        self.mlp_trans = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.ReLU(),
            nn.Linear(graph_dim, graph_dim)
        )

        # Attention projections
        self.w_q = nn.Linear(graph_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(graph_dim, attn_dim, bias=False)
        self.attn_vec = nn.Parameter(torch.randn(attn_dim * 2))

        # Dynamic benefit gate
        # Input: [s_ij (1), Ue(j) (1), selection_density (1)] -> 3-dim
        self.w_gate = nn.Linear(3, 1)

        # Recurrent node update in 64-d
        self.gru_update = nn.GRUCell(graph_dim, graph_dim)

        # Feature expansion from 64 -> 256 to match visual features
        self.expand_proj = nn.Sequential(
            nn.Linear(graph_dim, out_channels),
            nn.LayerNorm(out_channels)
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        m_st: torch.Tensor,
        s_ij: torch.Tensor,
        ue_flat: torch.Tensor,
        scale_splits: List[int],
        spatial_shapes: List[Tuple[int, int]]
    ) -> List[torch.Tensor]:
        """Performs benefit-aware message passing and returns [H1, H2, H3] with C=256."""
        device = nodes.device
        num_nodes = nodes.shape[0]
        src, tgt = edge_index[0], edge_index[1]

        # 1. Message transformation
        msg_val = self.mlp_trans(nodes[tgt])  # [E, 64]

        # 2. Attention scoring
        q = self.w_q(nodes[src])
        k = self.w_k(nodes[tgt])
        cat_qk = torch.cat([q, k], dim=-1)
        raw_attn = F.leaky_relu(torch.matmul(cat_qk, self.attn_vec), negative_slope=0.2)

        # Masked post-softmax normalization
        # Active edges where m_st > 0
        exp_attn = torch.exp(raw_attn) * m_st
        src_attn_sum = torch.zeros(num_nodes, device=device, dtype=nodes.dtype)
        src_attn_sum.scatter_add_(0, src, exp_attn)
        alpha_ij = exp_attn / (src_attn_sum[src] + 1e-8)  # [E]

        # 3. Dynamic benefit message gate
        # Local routing selection density rho_i
        src_edge_count = torch.zeros(num_nodes, device=device, dtype=nodes.dtype)
        src_edge_count.scatter_add_(0, src, m_st)
        rho_i = (src_edge_count[src] / float(self.k_total)).clamp(0.0, 1.0)

        gate_in = torch.stack([s_ij, ue_flat[tgt], rho_i], dim=-1)
        g_ij = torch.sigmoid(self.w_gate(gate_in)).squeeze(-1)  # [E]

        # 4. Message aggregation with explicit ST multiplication
        # m_st guarantees Straight-Through gradient flow into s_ij!
        weighted_msg = m_st.unsqueeze(1) * g_ij.unsqueeze(1) * alpha_ij.unsqueeze(1) * msg_val
        agg_msgs = torch.zeros_like(nodes)
        agg_msgs.scatter_add_(0, src.unsqueeze(1).repeat(1, self.graph_dim), weighted_msg)

        # 5. GRU node update in 64-d
        updated_nodes = self.gru_update(agg_msgs, nodes)  # [Num_nodes, 64]

        # 6. Feature expansion to 256-d
        expanded_nodes = self.expand_proj(updated_nodes)  # [Num_nodes, 256]

        # 7. Reshape back to spatial feature pyramids [H1, H2, H3]
        h_pyramid = []
        curr_offset = 0
        for n_nodes, (h_k, w_k) in zip(scale_splits, spatial_shapes):
            scale_feats = expanded_nodes[curr_offset:curr_offset + n_nodes]
            # Reshape [H*W, 256] -> [1, 256, H, W]
            scale_map = scale_feats.reshape(h_k, w_k, self.out_channels).permute(2, 0, 1).unsqueeze(0)
            h_pyramid.append(scale_map)
            curr_offset += n_nodes

        return h_pyramid

