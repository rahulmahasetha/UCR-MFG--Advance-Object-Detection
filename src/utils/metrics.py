from typing import List, Dict, Any, Tuple
import numpy as np
import torch


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU matrix between boxes1 [N, 4] and boxes2 [M, 4]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / np.maximum(union, 1e-7)


def compute_dece(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    num_classes: int = 10,
    iou_thresh: float = 0.5,
    num_bins: int = 15
) -> float:
    """Same-Class Detection Expected Calibration Error (D-ECE) including False Positives.
    
    Duplicates and background false alarms are labeled False Positive (y = 0).
    Unmatched predictions inside ignore masks are excluded.
    """
    all_scores: List[float] = []
    all_correct: List[int] = []

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred.get("boxes", np.empty((0, 4)))
        pred_scores = pred.get("scores", np.empty((0,)))
        pred_labels = pred.get("labels", np.empty((0,), dtype=int))

        gt_boxes = gt.get("boxes", np.empty((0, 4)))
        gt_labels = gt.get("labels", np.empty((0,), dtype=int))
        ignore_mask = gt.get("ignore_mask", None)

        if len(pred_boxes) == 0:
            continue

        # Sort predictions descending by score
        sort_idx = np.argsort(-pred_scores)
        p_boxes = pred_boxes[sort_idx]
        p_scores = pred_scores[sort_idx]
        p_labels = pred_labels[sort_idx]

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        for i in range(len(p_boxes)):
            score = float(p_scores[i])
            c = int(p_labels[i])
            box = p_boxes[i]

            # Same-class matching only
            c_gt_indices = np.where(gt_labels == c)[0]
            matched = False

            if len(c_gt_indices) > 0:
                c_gt_boxes = gt_boxes[c_gt_indices]
                ious = compute_iou_matrix(box[None, :], c_gt_boxes)[0]
                best_gt_sub_idx = np.argmax(ious)
                best_iou = ious[best_gt_sub_idx]
                real_gt_idx = c_gt_indices[best_gt_sub_idx]

                if best_iou >= iou_thresh:
                    if not gt_matched[real_gt_idx]:
                        # True Positive
                        gt_matched[real_gt_idx] = True
                        all_scores.append(score)
                        all_correct.append(1)
                        matched = True
                    else:
                        # Duplicate detection (False Positive)
                        all_scores.append(score)
                        all_correct.append(0)
                        matched = True

            if not matched:
                # Check if center falls inside ignore mask
                cx = int(round((box[0] + box[2]) / 2.0))
                cy = int(round((box[1] + box[3]) / 2.0))
                is_ignored = False
                if ignore_mask is not None:
                    h, w = ignore_mask.shape
                    if 0 <= cy < h and 0 <= cx < w and ignore_mask[cy, cx] == 1:
                        is_ignored = True

                if not is_ignored:
                    # Background False Alarm (False Positive)
                    all_scores.append(score)
                    all_correct.append(0)

    if not all_scores:
        return 0.0

    scores_arr = np.array(all_scores)
    correct_arr = np.array(all_correct)
    n_eval = len(scores_arr)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    dece = 0.0

    for m in range(num_bins):
        low, high = bin_edges[m], bin_edges[m + 1]
        in_bin = (scores_arr > low) & (scores_arr <= high) if m > 0 else (scores_arr >= low) & (scores_arr <= high)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            acc_bm = np.mean(correct_arr[in_bin])
            conf_bm = np.mean(scores_arr[in_bin])
            dece += (bin_count / n_eval) * abs(acc_bm - conf_bm)

    return float(dece)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """11-point interpolated Average Precision."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

