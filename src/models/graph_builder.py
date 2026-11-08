from typing import Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleGraphBuilder(nn.Module):
    """Constructs multi-scale candidate graph with spatial block partitioning and Student-t similarity."""

    def __init__(
        self,
        in_channels: int = 256,
        graph_dim: int = 64,
        r_local: int = 3,
        k_feat: int = 8,
        alpha: float = 1.0,
        eps: float = 1e-8,
        f1_block_size: int = 32
    ):
        super().__init__()
        self.graph_dim = graph_dim
        self.r_local = r_local
        self.k_feat = k_feat
        self.alpha = alpha
        self.eps = eps
        self.f1_block_size = f1_block_size

        # Node projection: 256 -> 64
        self.node_proj = nn.Sequential(
            nn.Linear(in_channels, graph_dim),
            nn.LayerNorm(graph_dim)
        )

    def project_nodes(self, f: torch.Tensor) -> torch.Tensor:
        """Projects feature map [B, C, H, W] to [B, N, graph_dim] where N = H * W."""
        b, c, h, w = f.shape
        f_flat = f.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return self.node_proj(f_flat)

    @torch.no_grad()
    def build_local_edges(self, h: int, w: int, offset: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds local spatial window edges within radius r_local."""
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        y_flat = y.reshape(-1)
        x_flat = x.reshape(-1)
        num_nodes = h * w

        sources, targets, deltas = [], [], []
        for dy in range(-self.r_local, self.r_local + 1):
            for dx in range(-self.r_local, self.r_local + 1):
                if dx == 0 and dy == 0:
                    continue
                ny = y_flat + dy
                nx = x_flat + dx
                valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
                src = torch.nonzero(valid).squeeze(1)
                tgt = ny[valid] * w + nx[valid]

                sources.append(src + offset)
                targets.append(tgt + offset)

                dist = torch.sqrt(torch.tensor(dx**2 + dy**2, device=device, dtype=torch.float32))
                delta = torch.tensor([float(dx), float(dy), torch.log(1.0 + dist)], device=device).unsqueeze(0).repeat(len(src), 1)
                deltas.append(delta)

        if not sources:
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 3), dtype=torch.float32, device=device)

        edge_index = torch.stack([torch.cat(sources), torch.cat(targets)], dim=0)
        edge_deltas = torch.cat(deltas, dim=0)
        return edge_index, edge_deltas

    @torch.no_grad()
    def build_knn_edges(
        self,
        node_feats: torch.Tensor,
        h: int,
        w: int,
        offset: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes feature-space kNN edges with spatial block partitioning on high-res grids."""
        device = node_feats.device
        norm_feats = F.normalize(node_feats, p=2, dim=-1)

        # Spatial block partitioning
        bs = self.f1_block_size
        if h > bs or w > bs:
            sources, targets = [], []
            for by in range(0, h, bs):
                for bx in range(0, w, bs):
                    # Get node indices in block
                    y_indices = torch.arange(by, min(by + bs, h), device=device)
                    x_indices = torch.arange(bx, min(bx + bs, w), device=device)
                    grid_y, grid_x = torch.meshgrid(y_indices, x_indices, indexing="ij")
                    block_nodes = (grid_y * w + grid_x).reshape(-1)

                    block_feats = norm_feats[block_nodes]
                    sim = torch.mm(block_feats, block_feats.t())
                    sim.fill_diagonal_(-float("inf"))
                    k = min(self.k_feat, len(block_nodes) - 1)
                    if k <= 0:
                        continue
                    _, topk_idx = torch.topk(sim, k=k, dim=1)

                    src_b = block_nodes.unsqueeze(1).repeat(1, k).reshape(-1)
                    tgt_b = block_nodes[topk_idx.reshape(-1)]
                    sources.append(src_b + offset)
                    targets.append(tgt_b + offset)

            if not sources:
                return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 3), dtype=torch.float32, device=device)

            edge_index = torch.stack([torch.cat(sources), torch.cat(targets)], dim=0)
        else:
            sim = torch.mm(norm_feats, norm_feats.t())
            sim.fill_diagonal_(-float("inf"))
            k = min(self.k_feat, norm_feats.shape[0] - 1)
            _, topk_idx = torch.topk(sim, k=k, dim=1)

            src = torch.arange(norm_feats.shape[0], device=device).unsqueeze(1).repeat(1, k).reshape(-1)
            tgt = topk_idx.reshape(-1)
            edge_index = torch.stack([src + offset, tgt + offset], dim=0)

        # Spatial geometric delta for kNN
        deltas = torch.zeros((edge_index.shape[1], 3), device=device, dtype=torch.float32)
        return edge_index, deltas

    @torch.no_grad()
    def build_cross_scale_edges(
        self,
        h1: int, w1: int, offset1: int,
        h2: int, w2: int, offset2: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds bidirectional cross-scale parent/child edges."""
        y2, x2 = torch.meshgrid(torch.arange(h2, device=device), torch.arange(w2, device=device), indexing="ij")
        y2_flat = y2.reshape(-1)
        x2_flat = x2.reshape(-1)

        # Scale 2 to Scale 1 (stride 2x parent-child)
        y1_proj = (y2_flat * 2).clamp(0, h1 - 1)
        x1_proj = (x2_flat * 2).clamp(0, w1 - 1)

        src_p2c = y2_flat * w2 + x2_flat + offset2
        tgt_p2c = y1_proj * w1 + x1_proj + offset1

        # Child to parent
        src_c2p = tgt_p2c
        tgt_c2p = src_p2c

        edge_index = torch.stack([
            torch.cat([src_p2c, src_c2p]),
            torch.cat([tgt_p2c, tgt_c2p])
        ], dim=0)

        deltas = torch.zeros((edge_index.shape[1], 3), device=device, dtype=torch.float32)
        return edge_index, deltas

    def compute_student_t_weights(self, edge_index: torch.Tensor, all_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Student-t similarity kernel kij and normalized edge weights wij."""
        src, tgt = edge_index[0], edge_index[1]
        diff = all_nodes[src] - all_nodes[tgt]
        sq_dist = torch.sum(diff ** 2, dim=-1)

        kij = torch.pow(1.0 + sq_dist / self.alpha, -(self.alpha + 1.0) / 2.0)

        # Softmax normalization per source node
        # Safe scatter-sum normalization
        src_max = torch.zeros(all_nodes.shape[0], device=all_nodes.device, dtype=all_nodes.dtype)
        src_max.scatter_reduce_(0, src, kij, reduce="amax", include_self=False)
        exp_k = torch.exp(kij - src_max[src])

        src_sum = torch.zeros(all_nodes.shape[0], device=all_nodes.device, dtype=all_nodes.dtype)
        src_sum.scatter_add_(0, src, exp_k)
        wij = exp_k / (src_sum[src] + self.eps)

        return kij, wij

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """Constructs candidate graphs for a batch of multi-scale pyramid features."""
        batch_size = features[0].shape[0]
        h1, w1 = features[0].shape[-2:]
        h2, w2 = features[1].shape[-2:]
        h3, w3 = features[2].shape[-2:]
        device = features[0].device

        # Node counts and offsets per scale
        n1 = h1 * w1
        n2 = h2 * w2
        n3 = h3 * w3
        off1 = 0
        off2 = n1
        off3 = n1 + n2

        # Project all scales to 64-d
        proj1 = self.project_nodes(features[0])  # [B, N1, 64]
        proj2 = self.project_nodes(features[1])  # [B, N2, 64]
        proj3 = self.project_nodes(features[2])  # [B, N3, 64]

        batch_graphs = []
        for b in range(batch_size):
            nodes_b = torch.cat([proj1[b], proj2[b], proj3[b]], dim=0)  # [Total_nodes, 64]

            # 1. Local window edges per scale
            e_loc1, d_loc1 = self.build_local_edges(h1, w1, off1, device)
            e_loc2, d_loc2 = self.build_local_edges(h2, w2, off2, device)
            e_loc3, d_loc3 = self.build_local_edges(h3, w3, off3, device)

            # 2. Feature kNN edges per scale
            e_knn1, d_knn1 = self.build_knn_edges(proj1[b], h1, w1, off1)
            e_knn2, d_knn2 = self.build_knn_edges(proj2[b], h2, w2, off2)
            e_knn3, d_knn3 = self.build_knn_edges(proj3[b], h3, w3, off3)

            # 3. Cross-scale edges (F1 <-> F2 and F2 <-> F3)
            e_cross12, d_cross12 = self.build_cross_scale_edges(h1, w1, off1, h2, w2, off2, device)
            e_cross23, d_cross23 = self.build_cross_scale_edges(h2, w2, off2, h3, w3, off3, device)

            # Edge type tags: 0=local, 1=knn, 2=cross_down, 3=cross_up
            type_loc = torch.zeros(e_loc1.shape[1] + e_loc2.shape[1] + e_loc3.shape[1], dtype=torch.long, device=device)
            type_knn = torch.ones(e_knn1.shape[1] + e_knn2.shape[1] + e_knn3.shape[1], dtype=torch.long, device=device)
            type_cross = torch.full((e_cross12.shape[1] + e_cross23.shape[1],), 2, dtype=torch.long, device=device)

            all_edges = torch.cat([e_loc1, e_loc2, e_loc3, e_knn1, e_knn2, e_knn3, e_cross12, e_cross23], dim=1)
            all_deltas = torch.cat([d_loc1, d_loc2, d_loc3, d_knn1, d_knn2, d_knn3, d_cross12, d_cross23], dim=0)
            all_types = torch.cat([type_loc, type_knn, type_cross], dim=0)

            # Deduplication with local precedence
            # Pack src, tgt into 64-bit keys
            keys = (all_edges[0].long() << 32) | all_edges[1].long()
            unique_keys, inv_idx = torch.unique(keys, return_inverse=True)
            perm = torch.arange(keys.size(0), dtype=torch.long, device=device)
            first_idx = torch.empty(unique_keys.size(0), dtype=torch.long, device=device)
            first_idx.scatter_(0, inv_idx, perm)

            dedup_edges = all_edges[:, first_idx]
            dedup_deltas = all_deltas[first_idx]
            dedup_types = all_types[first_idx]

            kij, wij = self.compute_student_t_weights(dedup_edges, nodes_b)

            batch_graphs.append({
                "nodes": nodes_b,
                "edge_index": dedup_edges,
                "edge_types": dedup_types,
                "edge_deltas": dedup_deltas,
                "kij": kij,
                "wij": wij,
                "scale_splits": [n1, n2, n3],
            })

        return batch_graphs
