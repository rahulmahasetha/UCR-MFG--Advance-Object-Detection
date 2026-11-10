import torch
import torch.nn as nn

from src.models.graph_ops import Graph


class GraphLaplacianLoss(nn.Module):
    """Dirichlet energy sum_ij w_ij ||h_i - h_j||^2 of the candidate graph, averaged over edges."""

    def forward(self, graph: Graph) -> torch.Tensor:
        if graph.edge_index.shape[1] == 0:
            return graph.nodes.sum() * 0.0
        recv, send = graph.edge_index
        sq_dist = (graph.nodes[recv] - graph.nodes[send]).float().pow(2).sum(-1)
        return (graph.wij * sq_dist).mean()
