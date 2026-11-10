from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
AREA_RANGES = {"all": (0.0, np.inf), "small": (0.0, 32.0 ** 2), "medium": (32.0 ** 2, 96.0 ** 2), "large": (96.0 ** 2, np.inf)}


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


def _area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _center_ignored(boxes: np.ndarray, ignore_mask: Optional[np.ndarray]) -> np.ndarray:
    if ignore_mask is None or len(boxes) == 0:
        return np.zeros(len(boxes), dtype=bool)
    h, w = ignore_mask.shape
    cx = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2).astype(int), 0, w - 1)
    cy = np.clip(((boxes[:, 1] + boxes[:, 3]) / 2).astype(int), 0, h - 1)
    return ignore_mask[cy, cx] > 0


def match_detections(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, gt_ignore: np.ndarray, thresholds: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy COCO matching of score-sorted predictions at every IoU threshold.

    Returns (tp, ignored), each [T, P]; matches to ignored GT are neither TP nor FP.
    """
    thresholds = np.asarray(thresholds)
    num_t, num_p = len(thresholds), len(pred_boxes)
    tp = np.zeros((num_t, num_p), dtype=bool)
    ignored = np.zeros((num_t, num_p), dtype=bool)
    if num_p == 0 or len(gt_boxes) == 0:
        return tp, ignored

    ious = compute_iou_matrix(pred_boxes, gt_boxes)
    matched = np.zeros((num_t, len(gt_boxes)), dtype=bool)
    priority = np.where(gt_ignore, 0.0, 2.0)
    rows = np.arange(num_t)
    for i in range(num_p):
        ok = (ious[i][None] >= thresholds[:, None]) & ~matched
        rank = np.where(ok, ious[i][None] + priority[None], -1.0)
        best = rank.argmax(axis=1)
        hit = rank[rows, best] >= 0
        matched[rows[hit], best[hit]] = True
        tp[hit, i] = ~gt_ignore[best[hit]]
        ignored[hit, i] = gt_ignore[best[hit]]
    return tp, ignored


def _average_precision(scores: np.ndarray, tp: np.ndarray, ignored: np.ndarray, num_gt: int) -> np.ndarray:
    if num_gt == 0:
        return np.full(tp.shape[0], np.nan)
    if scores.size == 0:
        return np.zeros(tp.shape[0])
    order = np.argsort(-scores, kind="mergesort")
    tp, ignored = tp[:, order], ignored[:, order]
    ctp = np.cumsum(tp, axis=1)
    cfp = np.cumsum(~tp & ~ignored, axis=1)
    recall = ctp / num_gt
    precision = np.maximum.accumulate((ctp / np.maximum(ctp + cfp, 1e-12))[:, ::-1], axis=1)[:, ::-1]

    recall_points = np.linspace(0.0, 1.0, 101)
    ap = np.zeros(tp.shape[0])
    for t in range(tp.shape[0]):
        idx = np.searchsorted(recall[t], recall_points, side="left")
        valid = idx < recall.shape[1]
        ap[t] = np.where(valid, precision[t, np.minimum(idx, recall.shape[1] - 1)], 0.0).mean()
    return ap


def compute_map(
    predictions: List[Dict[str, np.ndarray]], ground_truths: List[Dict[str, Any]], num_classes: int
) -> Dict[str, Any]:
    """COCO-style AP over IoU 0.50:0.95 (101-point), with area breakdown and VisDrone ignore regions."""
    per_area = {}
    for name, (lo, hi) in AREA_RANGES.items():
        ap = np.full((num_classes, len(IOU_THRESHOLDS)), np.nan)
        for c in range(num_classes):
            scores, tps, igns, num_gt = [], [], [], 0
            for pred, gt in zip(predictions, ground_truths):
                pm = pred["labels"] == c
                order = np.argsort(-pred["scores"][pm], kind="mergesort")
                p_boxes, p_scores = pred["boxes"][pm][order], pred["scores"][pm][order]
                g_boxes = gt["boxes"][gt["labels"] == c]
                g_area = _area(g_boxes)
                g_ignore = (g_area < lo) | (g_area >= hi)
                num_gt += int((~g_ignore).sum())

                tp, ign = match_detections(p_boxes, g_boxes, g_ignore, IOU_THRESHOLDS)
                p_area = _area(p_boxes)
                drop = (p_area < lo) | (p_area >= hi) | _center_ignored(p_boxes, gt.get("ignore_mask"))
                ign |= ~tp & ~ign & drop[None]
                scores.append(p_scores)
                tps.append(tp)
                igns.append(ign)
            ap[c] = _average_precision(np.concatenate(scores), np.concatenate(tps, 1), np.concatenate(igns, 1), num_gt)
        per_area[name] = ap

    def mean(x: np.ndarray) -> float:
        return float(np.nanmean(x)) if np.isfinite(x).any() else 0.0

    full = per_area["all"]
    return {
        "mAP": mean(full),
        "AP50": mean(full[:, 0]),
        "AP75": mean(full[:, 5]),
        "APs": mean(per_area["small"]),
        "APm": mean(per_area["medium"]),
        "APl": mean(per_area["large"]),
        "per_class_AP": [mean(full[c]) for c in range(num_classes)],
    }


def uncertainty_auroc(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, Any]],
    keys: Sequence[str] = ("u_a", "u_e", "box_std"),
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """AUROC of each uncertainty for separating false from true detections (higher = more informative).
    The confidence baseline uses 1 - score."""
    errors, values = [], {k: [] for k in (*keys, "1-score")}
    for pred, gt in zip(predictions, ground_truths):
        is_tp = np.zeros(len(pred["scores"]), dtype=bool)
        is_ign = np.zeros_like(is_tp)
        for c in np.unique(pred["labels"]):
            idx = np.where(pred["labels"] == c)[0]
            idx = idx[np.argsort(-pred["scores"][idx], kind="mergesort")]
            g_boxes = gt["boxes"][gt["labels"] == c]
            tp, ign = match_detections(pred["boxes"][idx], g_boxes, np.zeros(len(g_boxes), bool), [iou_thresh])
            is_tp[idx], is_ign[idx] = tp[0], ign[0]
        keep = is_tp | ~_center_ignored(pred["boxes"], gt.get("ignore_mask"))
        errors.append(~is_tp[keep])
        for k in keys:
            v = pred[k]
            values[k].append((v.mean(axis=1) if v.ndim > 1 else v)[keep])
        values["1-score"].append(1.0 - pred["scores"][keep])

    y = np.concatenate(errors) if errors else np.zeros(0, bool)
    if y.size == 0 or y.all() or not y.any():
        return {k: float("nan") for k in values}
    return {k: float(roc_auc_score(y, np.concatenate(v))) for k, v in values.items()}
