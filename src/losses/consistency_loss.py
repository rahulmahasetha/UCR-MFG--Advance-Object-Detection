from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectCentricConsistencyLoss(nn.Module):
    """Object-Centric Cross-Scale Prediction-Space KL Consistency Loss.
    
    Computes KL divergence between parent and child scales for ground-truth matched object centers.
    Implements minimal-area collision resolution and NaN-safe zero-pair autograd attachment.
    """

    def __init__(self, strides: List[int] = [8, 16, 32]):
        super().__init__()
        self.strides = strides

    def forward(
        self,
        cls_preds: List[torch.Tensor],
        gt_boxes_batch: List[torch.Tensor]
    ) -> torch.Tensor:
        device = cls_preds[0].device
        batch_size = cls_preds[0].shape[0]
        kl_losses = []

        # Compare scale k with scale k+1: (F1, F2) and (F2, F3)
        for lvl in range(len(self.strides) - 1):
            s_k = self.strides[lvl]
            s_k1 = self.strides[lvl + 1]

            pred_k = cls_preds[lvl]    # [B, C, H_k, W_k]
            pred_k1 = cls_preds[lvl + 1]  # [B, C, H_k1, W_k1]
            c_dim = pred_k.shape[1]

            for b in range(batch_size):
                boxes = gt_boxes_batch[b].to(device)
                if boxes.numel() == 0:
                    continue

                # Ground-truth object centers
                gx = (boxes[:, 0] + boxes[:, 2]) / 2.0
                gy = (boxes[:, 1] + boxes[:, 3]) / 2.0
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

                # Projected grid locations
                xk = (gx / s_k).long().clamp(0, pred_k.shape[3] - 1)
                yk = (gy / s_k).long().clamp(0, pred_k.shape[2] - 1)
                xk1 = (gx / s_k1).long().clamp(0, pred_k1.shape[3] - 1)
                yk1 = (gy / s_k1).long().clamp(0, pred_k1.shape[2] - 1)

                # Minimal-area collision resolution for level k
                keys_k = (yk * pred_k.shape[3] + xk).long()
                unique_keys, inv_idx = torch.unique(keys_k, return_inverse=True)

                matched_pairs = []
                for u_idx in range(unique_keys.size(0)):
                    matching_gt = torch.nonzero(inv_idx == u_idx).squeeze(1)
                    if matching_gt.numel() == 1:
                        best_gt = matching_gt[0]
                    else:
                        best_gt = matching_gt[areas[matching_gt].argmin()]

                    matched_pairs.append((
                        (yk[best_gt], xk[best_gt]),
                        (yk1[best_gt], xk1[best_gt])
                    ))

                if not matched_pairs:
                    continue

                # Extract logits at matched locations
                logits_child = torch.stack([pred_k[b, :, y_c, x_c] for (y_c, x_c), _ in matched_pairs], dim=0)
                logits_parent = torch.stack([pred_k1[b, :, y_p, x_p] for _, (y_p, x_p) in matched_pairs], dim=0)

                log_p_child = F.log_softmax(logits_child, dim=-1)
                p_parent = F.softmax(logits_parent.detach(), dim=-1)  # Stop grad into parent

                # KL divergence: D_KL(child || parent)
                kl = F.kl_div(log_p_child, p_parent, reduction="batchmean")
                kl_losses.append(kl)

        # NaN-safe empty set handling: Attach 0.0 * tensor sum to maintain autograd
        if not kl_losses:
            zero_grad = 0.0 * sum([p.sum() for p in cls_preds])
            return zero_grad

        return sum(kl_losses) / len(kl_losses)

