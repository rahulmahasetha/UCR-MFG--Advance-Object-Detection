from typing import List, Dict, Any
import torch
import torch.nn as nn


class GraphLaplacianLoss(nn.Module):
    """Laplacian Dirichlet Energy loss over 64-d candidate graph to enforce feature smoothness."""

    def __init__(self):
        super().__init__()

    def forward(self, graph_data_list: List[Dict[str, Any]]) -> torch.Tensor:
        losses = []
        for g_dict in graph_data_list:
            nodes = g_dict["nodes"]
            edge_index = g_dict["edge_index"]
            wij = g_dict["wij"]

            if edge_index.shape[1] == 0:
                continue

            src, tgt = edge_index[0], edge_index[1]
            diff_sq = torch.sum((nodes[src] - nodes[tgt]) ** 2, dim=-1)
            e_loss = (wij * diff_sq).mean()
            losses.append(e_loss)

        if not losses:
            return torch.tensor(0.0, device=graph_data_list[0]["nodes"].device)

        return sum(losses) / len(losses)

