import math
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCOSHead(nn.Module):
    """Downstream anchor-free FCOS detection head with decoupled Cls, Reg, Centerness towers."""

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        strides: List[int] = [8, 16, 32],
        center_sampling_radius: float = 1.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.center_radius = center_sampling_radius
        self.scale_limits = [[0, 64], [64, 128], [128, float("inf")]]

        # Classification tower (4 convs)
        cls_layers = []
        for _ in range(4):
            cls_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            cls_layers.append(nn.GroupNorm(32, in_channels))
            cls_layers.append(nn.SiLU())
        self.cls_tower = nn.Sequential(*cls_layers)

        # Regression tower (4 convs)
        reg_layers = []
        for _ in range(4):
            reg_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            reg_layers.append(nn.GroupNorm(32, in_channels))
            reg_layers.append(nn.SiLU())
        self.reg_tower = nn.Sequential(*reg_layers)

        # Prediction heads
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.cen_pred = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # Scale multipliers per FPN level for regression
        self.scales = nn.ModuleList([ScaleModule(init_value=1.0) for _ in strides])

        # Prior bias initialization for classification (-4.6 => p ~ 0.01)
        prior_prob = 0.01
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_pred.bias, bias_init)

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass over feature pyramid [Z1, Z2, Z3].
        
        Returns:
            cls_preds: List of [B, C, H_k, W_k] logits
            reg_preds: List of [B, 4, H_k, W_k] stride-normalized distances
            cen_preds: List of [B, 1, H_k, W_k] centerness logits
        """
        cls_preds, reg_preds, cen_preds = [], [], []

        for lvl, feat in enumerate(features):
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            cls_logits = self.cls_pred(cls_feat)
            cen_logits = self.cen_pred(cls_feat)

            # Stride-normalized positive regression predictions
            raw_reg = self.reg_pred(reg_feat)
            scaled_reg = self.scales[lvl](raw_reg)
            reg_dist = torch.exp(scaled_reg.clamp(-10.0, 10.0))

            cls_preds.append(cls_logits)
            reg_preds.append(reg_dist)
            cen_preds.append(cen_logits)

        return cls_preds, reg_preds, cen_preds

    @torch.no_grad()
    def get_grid_locations(
        self,
        shapes: List[Tuple[int, int]],
        device: torch.device
    ) -> List[torch.Tensor]:
        """Generates pixel center coordinates (cx, cy) for all feature pyramid locations."""
        locations = []
        for stride, (h, w) in zip(self.strides, shapes):
            shift_y = torch.arange(0, h, device=device) * stride + stride / 2.0
            shift_x = torch.arange(0, w, device=device) * stride + stride / 2.0
            grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            loc = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
            locations.append(loc)
        return locations

    @torch.no_grad()
    def assign_targets(
        self,
        locations: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        ignore_masks: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Center-sampling target assignment adhering strictly to FCOS scale limits and ignore masks."""
        batch_size = len(gt_boxes)
        device = locations[0].device

        cls_targets_all, reg_targets_all, cen_targets_all, pos_masks_all = [], [], [], []

        for lvl, (loc, stride, (min_scale, max_scale)) in enumerate(zip(locations, self.strides, self.scale_limits)):
            lvl_cls, lvl_reg, lvl_cen, lvl_pos = [], [], [], []
            n_loc = loc.shape[0]

            for b in range(batch_size):
                boxes_b = gt_boxes[b].to(device)
                labels_b = gt_labels[b].to(device)
                ignore_b = ignore_masks[b].to(device)

                cls_target = torch.full((n_loc,), -1, dtype=torch.long, device=device)
                reg_target = torch.zeros((n_loc, 4), dtype=torch.float32, device=device)
                cen_target = torch.zeros((n_loc,), dtype=torch.float32, device=device)
                pos_mask = torch.zeros((n_loc,), dtype=torch.bool, device=device)

                if boxes_b.numel() == 0:
                    lvl_cls.append(cls_target)
                    lvl_reg.append(reg_target)
                    lvl_cen.append(cen_target)
                    lvl_pos.append(pos_mask)
                    continue

                # Distances from location to box boundaries
                # loc: [N, 2], boxes: [M, 4]
                xs, ys = loc[:, 0].unsqueeze(1), loc[:, 1].unsqueeze(1)
                l = xs - boxes_b[:, 0].unsqueeze(0)
                t = ys - boxes_b[:, 1].unsqueeze(0)
                r = boxes_b[:, 2].unsqueeze(0) - xs
                b_dist = boxes_b[:, 3].unsqueeze(0) - ys
                reg_dist = torch.stack([l, t, r, b_dist], dim=-1)  # [N, M, 4]

                # 1. Condition: Inside center box
                center_x = (boxes_b[:, 0] + boxes_b[:, 2]) / 2.0
                center_y = (boxes_b[:, 1] + boxes_b[:, 3]) / 2.0
                c_l = xs - (center_x - self.center_radius * stride).unsqueeze(0)
                c_t = ys - (center_y - self.center_radius * stride).unsqueeze(0)
                c_r = (center_x + self.center_radius * stride).unsqueeze(0) - xs
                c_b = (center_y + self.center_radius * stride).unsqueeze(0) - ys
                inside_center = torch.stack([c_l, c_t, c_r, c_b], dim=-1).min(dim=-1)[0] > 0

                # 2. Condition: Inside actual box
                inside_box = reg_dist.min(dim=-1)[0] > 0

                # 3. Condition: Within scale range
                max_reg = reg_dist.max(dim=-1)[0]
                in_scale = (max_reg >= min_scale) & (max_reg <= max_scale)

                valid_match = inside_center & inside_box & in_scale  # [N, M]

                # Minimal area ambiguity resolution
                areas = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])).unsqueeze(0).repeat(n_loc, 1)
                areas[~valid_match] = float("inf")
                min_area, matched_gt_idx = areas.min(dim=1)

                is_pos = min_area < float("inf")

                # Filter locations falling inside ignore masks
                loc_x_int = loc[:, 0].long().clamp(0, ignore_b.shape[1] - 1)
                loc_y_int = loc[:, 1].long().clamp(0, ignore_b.shape[0] - 1)
                in_ignore = ignore_b[loc_y_int, loc_x_int] == 1

                is_pos = is_pos & (~in_ignore)

                pos_indices = torch.nonzero(is_pos).squeeze(1)
                if pos_indices.numel() > 0:
                    assigned_gt = matched_gt_idx[pos_indices]
                    cls_target[pos_indices] = labels_b[assigned_gt]

                    pos_reg = reg_dist[pos_indices, assigned_gt]
                    reg_target[pos_indices] = pos_reg / stride  # Stride-normalized targets

                    # Centerness target
                    l_val, t_val, r_val, b_val = pos_reg[:, 0], pos_reg[:, 1], pos_reg[:, 2], pos_reg[:, 3]
                    cen_target[pos_indices] = torch.sqrt(
                        (torch.min(l_val, r_val) / torch.max(l_val, r_val)) *
                        (torch.min(t_val, b_val) / torch.max(t_val, b_val))
                    )
                    pos_mask[pos_indices] = True

                # Negative locations outside ignore regions
                neg_indices = torch.nonzero((~is_pos) & (~in_ignore)).squeeze(1)
                cls_target[neg_indices] = self.num_classes  # Background class ID

                lvl_cls.append(cls_target)
                lvl_reg.append(reg_target)
                lvl_cen.append(cen_target)
                lvl_pos.append(pos_mask)

            cls_targets_all.append(torch.stack(lvl_cls, dim=0))
            reg_targets_all.append(torch.stack(lvl_reg, dim=0))
            cen_targets_all.append(torch.stack(lvl_cen, dim=0))
            pos_masks_all.append(torch.stack(lvl_pos, dim=0))

        return cls_targets_all, reg_targets_all, cen_targets_all, pos_masks_all


class ScaleModule(nn.Module):
    """Learnable per-level scale multiplier for box regression."""
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
