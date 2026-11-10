import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.graph_ops import segment_softmax


class BenefitAwareMPNN(nn.Module):
    """C3: Benefit-aware multi-scale message passing over routed edges.

    Messages from sender j to receiver i are weighted by masked attention and a benefit gate
    g_ij = sigmoid(f(h_i, v_j, s_ij, Ua_i, Ue_i, Ue_j)), so uncertain receivers can admit context while
    unhelpful messages are suppressed. Only the edges passed in are computed. The routing mask enters
    the attention normaliser, which gives every passed edge (selected or not) a gradient for its benefit.
    """

    def __init__(self, graph_dim: int = 64, out_channels: int = 256, use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.value = nn.Sequential(nn.Linear(graph_dim, graph_dim), nn.ReLU(), nn.Linear(graph_dim, graph_dim))
        self.att_recv = nn.Linear(graph_dim, 1, bias=False)
        self.att_send = nn.Linear(graph_dim, 1, bias=False)
        self.gate_recv = nn.Linear(graph_dim, 1)
        self.gate_msg = nn.Linear(graph_dim, 1, bias=False)
        self.gate_edge = nn.Linear(4, 1, bias=False)
        self.gru = nn.GRUCell(graph_dim, graph_dim)
        self.expand = nn.Sequential(nn.Linear(graph_dim, out_channels), nn.LayerNorm(out_channels))

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
        scores: torch.Tensor,
        ua: torch.Tensor,
        ue: torch.Tensor,
    ) -> torch.Tensor:
        recv, send = edge_index
        num_nodes = nodes.shape[0]
        values = self.value(nodes)

        logits = F.leaky_relu(self.att_recv(nodes)[recv] + self.att_send(nodes)[send], 0.2).squeeze(-1)
        weight = segment_softmax(logits, recv, num_nodes, weight=mask)

        if self.use_gate:
            edge_terms = torch.stack([scores, ua[recv], ue[recv], ue[send]], dim=1).to(nodes.dtype)
            gate = self.gate_recv(nodes)[recv] + self.gate_msg(values)[send] + self.gate_edge(edge_terms)
            weight = weight * torch.sigmoid(gate.squeeze(-1).float())

        messages = weight.unsqueeze(1).to(values.dtype) * values[send]
        aggregated = torch.zeros_like(values).index_add_(0, recv, messages)
        return self.expand(self.gru(aggregated, nodes))
