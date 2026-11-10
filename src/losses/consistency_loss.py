from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectCentricConsistencyLoss(nn.Module):
    """Cross-scale class-distribution KL at GT object centers between adjacent pyramid levels.

    When several objects share a fine-level cell the smallest one is used; the coarse level is the
    detached teacher. Returns an autograd-attached zero when no object is present.
    """

    def __init__(self, strides: List[int] = [8, 16, 32]):
        super().__init__()
        self.strides = strides

    def forward(self, cls_preds: List[torch.Tensor], gt_boxes_batch: List[torch.Tensor]) -> torch.Tensor:
        losses = []
        for lvl in range(len(self.strides) - 1):
            child_map, parent_map = cls_preds[lvl], cls_preds[lvl + 1]
            for b, boxes in enumerate(gt_boxes_batch):
                if boxes.numel() == 0:
                    continue
                boxes = boxes.to(child_map.device)
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                area = (boxes[:, 2:] - boxes[:, :2]).prod(-1)
                cx, cy = self._cells(center, self.strides[lvl], child_map)
                px, py = self._cells(center, self.strides[lvl + 1], parent_map)

                keys = cy * child_map.shape[3] + cx
                order = torch.argsort(area)
                unique, inverse = torch.unique(keys[order], return_inverse=True)
                first = torch.full((unique.numel(),), order.numel(), device=keys.device, dtype=torch.long)
                first.scatter_reduce_(0, inverse, torch.arange(order.numel(), device=keys.device), reduce="amin")
                pick = order[first]

                child = child_map[b, :, cy[pick], cx[pick]].T.float()
                parent = parent_map[b, :, py[pick], px[pick]].T.float().detach()
                losses.append(F.kl_div(F.log_softmax(child, -1), F.softmax(parent, -1), reduction="batchmean"))

        if not losses:
            return sum(p.sum() for p in cls_preds) * 0.0
        return sum(losses) / len(losses)

    @staticmethod
    def _cells(center: torch.Tensor, stride: int, fmap: torch.Tensor):
        x = (center[:, 0] / stride).long().clamp(0, fmap.shape[3] - 1)
        y = (center[:, 1] / stride).long().clamp(0, fmap.shape[2] - 1)
        return x, y
