from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.graph_ops import BOTTOM_UP, KNN, LOCAL, TOP_DOWN, Graph, flatten_levels, segment_softmax


def edge_geometry(dxy: torch.Tensor) -> torch.Tensor:
    """Bounded direction plus log distance, in receiver grid units."""
    dist = dxy.norm(dim=-1, keepdim=True)
    return torch.cat([dxy / (1.0 + dist), torch.log1p(dist)], dim=-1)


class MultiScaleGraphBuilder(nn.Module):
    """Candidate multi-scale fuzzy graph over pyramid locations.

    Candidates are local windows, block-partitioned feature kNN and parent/child cross-scale links,
    weighted by a heavy-tailed Student-t affinity so ambiguous pairs keep a small non-zero link.
    """

    def __init__(
        self,
        in_channels: int = 256,
        graph_dim: int = 64,
        strides: Sequence[int] = (8, 16, 32),
        r_local: int = 3,
        k_feat: int = 8,
        alpha: float = 1.0,
        block_size: int = 32,
    ):
        super().__init__()
        self.strides = tuple(strides)
        self.r_local = r_local
        self.k_feat = k_feat
        self.alpha = alpha
        self.block_size = block_size
        self.node_proj = nn.Sequential(nn.Linear(in_channels, graph_dim), nn.LayerNorm(graph_dim))
        self._static_cache: Dict[Tuple, Tuple] = {}

    def forward(self, features: Sequence[torch.Tensor]) -> Graph:
        batch_size = features[0].shape[0]
        device = features[0].device
        shapes = [tuple(f.shape[-2:]) for f in features]
        positions, offsets, (s_edges, s_types, s_geom) = self._static(shapes, device)
        n = positions.shape[0]

        nodes = self.node_proj(flatten_levels(features))
        shift = torch.arange(batch_size, device=device).view(1, -1, 1) * n
        edges = [(s_edges.unsqueeze(1) + shift).reshape(2, -1)]
        types = [s_types.repeat(batch_size)]
        geoms = [s_geom.repeat(batch_size, 1)]

        per_image = nodes.view(batch_size, n, -1)
        for (h, w), off in zip(shapes, offsets):
            e, g = self._knn(per_image[:, off:off + h * w], h, w, off, n)
            edges.append(e)
            types.append(torch.full((e.shape[1],), KNN, dtype=torch.long, device=device))
            geoms.append(g)

        edge_index = torch.cat(edges, dim=1)
        kij, wij = self._affinity(nodes, edge_index)
        return Graph(
            nodes=nodes,
            edge_index=edge_index,
            edge_types=torch.cat(types),
            edge_geom=torch.cat(geoms),
            kij=kij,
            wij=wij,
            positions=positions,
            shapes=shapes,
            batch_size=batch_size,
        )

    def _affinity(self, nodes: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sq_dist = (nodes[edge_index[0]] - nodes[edge_index[1]]).float().pow(2).sum(-1)
        kij = (1.0 + sq_dist / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        wij = segment_softmax(kij, edge_index[0], nodes.shape[0])
        return kij, wij

    @torch.no_grad()
    def _knn(self, x: torch.Tensor, h: int, w: int, offset: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, device = x.shape[0], x.device
        bh, bw = min(self.block_size, h), min(self.block_size, w)
        gh, gw = -(-h // bh), -(-w // bw)
        ids = torch.full((gh * bh, gw * bw), -1, dtype=torch.long, device=device)
        ids[:h, :w] = torch.arange(h * w, device=device).view(h, w)
        ids = ids.view(gh, bh, gw, bw).permute(0, 2, 1, 3).reshape(gh * gw, bh * bw)
        block = ids.shape[1]
        k = min(self.k_feat, block - 1)
        if k <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 3), device=device)

        feats = F.normalize(x.float(), dim=-1)[:, ids.clamp(min=0)]
        sim = feats @ feats.transpose(-1, -2)
        invalid = (ids < 0).unsqueeze(1) | torch.eye(block, dtype=torch.bool, device=device)
        sim = sim.masked_fill(invalid, -torch.inf)
        val, nbr = sim.topk(k, dim=-1)

        recv = ids.view(1, *ids.shape, 1).expand_as(nbr)
        send = torch.gather(ids.expand(batch_size, -1, -1), 2, nbr.flatten(2)).view_as(nbr)
        dx = send % w - recv % w
        dy = send // w - recv // w
        outside_window = (dx.abs() > self.r_local) | (dy.abs() > self.r_local)
        keep = (recv >= 0) & torch.isfinite(val) & outside_window

        base = torch.arange(batch_size, device=device).view(-1, 1, 1, 1) * n + offset
        edges = torch.stack([(recv + base)[keep], (send + base)[keep]])
        geom = edge_geometry(torch.stack([dx[keep], dy[keep]], dim=-1).float())
        return edges, geom

    def _static(self, shapes: List[Tuple[int, int]], device: torch.device) -> Tuple:
        key = (tuple(shapes), str(device))
        if key not in self._static_cache:
            self._static_cache[key] = self._build_static(shapes, device)
        return self._static_cache[key]

    def _build_static(self, shapes: List[Tuple[int, int]], device: torch.device) -> Tuple:
        offsets = [0]
        for h, w in shapes[:-1]:
            offsets.append(offsets[-1] + h * w)
        centers = [self._centers(h, w, s, device) for (h, w), s in zip(shapes, self.strides)]

        edges, types, geoms = [], [], []
        for (h, w), off in zip(shapes, offsets):
            e, g = self._local(h, w, off, device)
            edges.append(e)
            types.append(torch.full((e.shape[1],), LOCAL, dtype=torch.long, device=device))
            geoms.append(g)

        for lvl in range(len(shapes) - 1):
            (h2, w2), s1, s2 = shapes[lvl + 1], self.strides[lvl], self.strides[lvl + 1]
            child = torch.arange(shapes[lvl][0] * shapes[lvl][1], device=device)
            c_xy = centers[lvl]
            py = torch.div(c_xy[:, 1], s2, rounding_mode="floor").long().clamp(max=h2 - 1)
            px = torch.div(c_xy[:, 0], s2, rounding_mode="floor").long().clamp(max=w2 - 1)
            parent = py * w2 + px
            delta = centers[lvl + 1][parent] - c_xy
            child, parent = child + offsets[lvl], parent + offsets[lvl + 1]

            edges += [torch.stack([child, parent]), torch.stack([parent, child])]
            types += [
                torch.full_like(child, TOP_DOWN),
                torch.full_like(child, BOTTOM_UP),
            ]
            geoms += [edge_geometry(delta / s1), edge_geometry(-delta / s2)]

        static = (torch.cat(edges, dim=1), torch.cat(types), torch.cat(geoms))
        return torch.cat(centers), offsets, static

    def _local(self, h: int, w: int, offset: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.r_local
        span = torch.arange(-r, r + 1, device=device)
        dy, dx = torch.meshgrid(span, span, indexing="ij")
        dy, dx = dy.reshape(-1), dx.reshape(-1)
        nonzero = (dx != 0) | (dy != 0)
        dy, dx = dy[nonzero], dx[nonzero]

        node = torch.arange(h * w, device=device)
        ny = (node // w).unsqueeze(1) + dy
        nx = (node % w).unsqueeze(1) + dx
        valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        recv = node.unsqueeze(1).expand_as(ny)[valid] + offset
        send = (ny * w + nx)[valid] + offset
        dxy = torch.stack([dx.expand_as(ny)[valid], dy.expand_as(ny)[valid]], dim=-1).float()
        return torch.stack([recv, send]), edge_geometry(dxy)

    @staticmethod
    def _centers(h: int, w: int, stride: int, device: torch.device) -> torch.Tensor:
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        return (torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1).float() + 0.5) * stride
