from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

LOCAL, KNN, TOP_DOWN, BOTTOM_UP = 0, 1, 2, 3
NUM_EDGE_TYPES = 4


@dataclass
class Graph:
    """Batched candidate graph. Edges are (receiver, sender) pairs over `batch_size * num_nodes` nodes."""

    nodes: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    edge_geom: torch.Tensor
    kij: torch.Tensor
    wij: torch.Tensor
    positions: torch.Tensor
    shapes: List[Tuple[int, int]]
    batch_size: int

    @property
    def num_nodes(self) -> int:
        return self.positions.shape[0]


@dataclass
class Routing:
    scores: torch.Tensor
    mask: torch.Tensor
    selected: torch.Tensor


def segment_softmax(
    logits: torch.Tensor, index: torch.Tensor, size: int, weight: torch.Tensor = None, eps: float = 1e-12
) -> torch.Tensor:
    logits = logits.float()
    seg_max = torch.full((size,), -torch.inf, device=logits.device).scatter_reduce(
        0, index, logits, reduce="amax", include_self=True
    )
    exp = torch.exp(logits - seg_max[index])
    if weight is not None:
        exp = exp * weight.float()
    denom = torch.zeros(size, device=logits.device).index_add_(0, index, exp)
    return exp / (denom[index] + eps)


def flatten_levels(maps: Sequence[torch.Tensor]) -> torch.Tensor:
    """[B, C, H_l, W_l] per level -> [B * N, C] in graph node order."""
    flat = torch.cat([m.flatten(2) for m in maps], dim=2)
    return flat.permute(0, 2, 1).reshape(-1, flat.shape[1])


def to_pyramid(x: torch.Tensor, batch_size: int, shapes: Sequence[Tuple[int, int]]) -> List[torch.Tensor]:
    """[B * N, C] in graph node order -> [B, C, H_l, W_l] per level."""
    x = x.view(batch_size, -1, x.shape[-1])
    sizes = [h * w for h, w in shapes]
    return [
        chunk.permute(0, 2, 1).reshape(batch_size, -1, h, w)
        for chunk, (h, w) in zip(torch.split(x, sizes, dim=1), shapes)
    ]
