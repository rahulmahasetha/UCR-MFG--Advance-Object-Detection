from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

        if len(pred_boxes) == 0:
            continue

        # Sort predictions descending by score
        sort_idx = np.argsort(-pred_scores)
        p_boxes = pred_boxes[sort_idx]
        p_scores = pred_scores[sort_idx]
        p_labels = pred_labels[sort_idx]

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        center_ignored = ignored_flags(pred, gt)[sort_idx]

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
                if not center_ignored[i]:
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


def ignored_flags(pred: Dict[str, np.ndarray], gt: Dict[str, Any]) -> np.ndarray:
    """Whether each detection's centre lies in an ignore region (precomputed at inference when available)."""
    if "center_ignored" in pred:
        return pred["center_ignored"]
    return _center_ignored(pred["boxes"], gt.get("ignore_mask"))


def match_detections(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, gt_ignore: np.ndarray, thresholds: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy COCO matching of score-sorted predictions at every IoU threshold.

    Returns (tp, ignored, gt_index), each [T, P]; matches to ignored GT are neither TP nor FP and
    gt_index is -1 for unmatched predictions.
    """
    thresholds = np.asarray(thresholds)
    num_t, num_p = len(thresholds), len(pred_boxes)
    tp = np.zeros((num_t, num_p), dtype=bool)
    ignored = np.zeros((num_t, num_p), dtype=bool)
    gt_index = np.full((num_t, num_p), -1, dtype=np.int64)
    if num_p == 0 or len(gt_boxes) == 0:
        return tp, ignored, gt_index

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
        gt_index[hit, i] = best[hit]
    return tp, ignored, gt_index


def build_records(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, Any]],
    num_classes: int,
    area: Tuple[float, float] = (0.0, np.inf),
    gt_keep: Optional[Callable[[Dict[str, Any]], np.ndarray]] = None,
) -> List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]]:
    """Per-class, per-image matching records (scores, tp, ignored, num_gt) for fast AP and bootstrap.

    GT outside `area` or rejected by `gt_keep` is ignored rather than removed, so detections of it are
    neither rewarded nor penalised. Unmatched detections centred in an ignore region are also ignored.
    """
    lo, hi = area
    records = [[] for _ in range(num_classes)]
    for pred, gt in zip(predictions, ground_truths):
        keep_all = gt_keep(gt) if gt_keep is not None else np.ones(len(gt["labels"]), dtype=bool)
        in_ignore_region = ignored_flags(pred, gt)
        for c in range(num_classes):
            pm = np.where(pred["labels"] == c)[0]
            pm = pm[np.argsort(-pred["scores"][pm], kind="mergesort")]
            p_boxes = pred["boxes"][pm]
            gm = gt["labels"] == c
            g_area = _area(gt["boxes"][gm])
            g_ignore = (g_area < lo) | (g_area >= hi) | ~keep_all[gm]
            tp, ign, _ = match_detections(p_boxes, gt["boxes"][gm], g_ignore, IOU_THRESHOLDS)
            p_area = _area(p_boxes)
            drop = (p_area < lo) | (p_area >= hi) | in_ignore_region[pm]
            ign |= ~tp & ~ign & drop[None]
            records[c].append((pred["scores"][pm], tp, ign, int((~g_ignore).sum())))
    return records


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


def ap_from_records(records, image_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """[num_classes, T] AP (NaN for classes without GT) over all images or a (re)sampled index list."""
    out = []
    for per_image in records:
        chosen = per_image if image_indices is None else [per_image[i] for i in image_indices]
        out.append(_average_precision(
            np.concatenate([r[0] for r in chosen]),
            np.concatenate([r[1] for r in chosen], axis=1),
            np.concatenate([r[2] for r in chosen], axis=1),
            sum(r[3] for r in chosen),
        ))
    return np.stack(out)


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def recall_at(records, iou_index: int = 0) -> float:
    hits = sum(int(r[1][iou_index].sum()) for per_image in records for r in per_image)
    total = sum(r[3] for per_image in records for r in per_image)
    return hits / total if total else float("nan")


def compute_map(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, Any]],
    num_classes: int,
    gt_keep: Optional[Callable[[Dict[str, Any]], np.ndarray]] = None,
) -> Dict[str, Any]:
    """COCO-style AP over IoU 0.50:0.95 (101-point), with area breakdown and VisDrone ignore regions."""
    per_area = {
        name: ap_from_records(build_records(predictions, ground_truths, num_classes, rng, gt_keep))
        for name, rng in AREA_RANGES.items()
    }
    full = per_area["all"]
    return {
        "mAP": _nanmean(full),
        "AP50": _nanmean(full[:, 0]),
        "AP75": _nanmean(full[:, 5]),
        "APs": _nanmean(per_area["small"]),
        "APm": _nanmean(per_area["medium"]),
        "APl": _nanmean(per_area["large"]),
        "per_class_AP": [_nanmean(full[c]) for c in range(num_classes)],
    }


def occlusion_breakdown(
    predictions: List[Dict[str, np.ndarray]], ground_truths: List[Dict[str, Any]], num_classes: int
) -> Dict[str, Dict[str, float]]:
    """AP and recall@0.5 per VisDrone occlusion level (0 none, 1 partial 1-50%, 2 heavy 50-100%)."""
    if not all("occlusion" in gt for gt in ground_truths):
        return {}
    out = {}
    for level, name in enumerate(("none", "partial", "heavy")):
        records = build_records(predictions, ground_truths, num_classes, gt_keep=lambda gt, k=level: gt["occlusion"] == k)
        ap = ap_from_records(records)
        out[name] = {
            "mAP": _nanmean(ap), "AP50": _nanmean(ap[:, 0]), "recall50": recall_at(records),
            "num_gt": int(sum(r[3] for per_image in records for r in per_image)),
        }
    return out


def detection_outcomes(
    predictions: List[Dict[str, np.ndarray]], ground_truths: List[Dict[str, Any]], iou_thresh: float = 0.5
) -> List[Dict[str, np.ndarray]]:
    """Per detection: is_tp (same class, IoU >= thresh, first match), counted (not an ignored-region FP),
    and the matched GT index (-1 if none)."""
    outcomes = []
    for pred, gt in zip(predictions, ground_truths):
        n = len(pred["scores"])
        is_tp, is_ign = np.zeros(n, bool), np.zeros(n, bool)
        gt_index = np.full(n, -1, dtype=np.int64)
        for c in np.unique(pred["labels"]):
            idx = np.where(pred["labels"] == c)[0]
            idx = idx[np.argsort(-pred["scores"][idx], kind="mergesort")]
            gm = np.where(gt["labels"] == c)[0]
            tp, ign, gi = match_detections(pred["boxes"][idx], gt["boxes"][gm], np.zeros(len(gm), bool), [iou_thresh])
            is_tp[idx], is_ign[idx] = tp[0], ign[0]
            gt_index[idx] = np.where(gi[0] >= 0, gm[np.maximum(gi[0], 0)] if len(gm) else -1, -1)
        counted = is_tp | ~ignored_flags(pred, gt)
        outcomes.append({"is_tp": is_tp, "counted": counted, "gt_index": gt_index})
    return outcomes


def uncertainty_value(pred: Dict[str, np.ndarray], key: str) -> np.ndarray:
    if key == "1-score":
        return 1.0 - pred["scores"]
    v = pred[key]
    return v.mean(axis=1) if v.ndim > 1 else v


def uncertainty_auroc(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, Any]],
    keys: Sequence[str] = ("u_a", "u_e", "box_std", "1-score"),
    iou_thresh: float = 0.5,
    outcomes: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Dict[str, float]:
    """AUROC of each uncertainty for separating false from true detections (0.5 = chance)."""
    outcomes = outcomes or detection_outcomes(predictions, ground_truths, iou_thresh)
    errors = [~o["is_tp"][o["counted"]] for o in outcomes]
    y = np.concatenate(errors) if errors else np.zeros(0, bool)
    if y.size == 0 or y.all() or not y.any():
        return {k: float("nan") for k in keys}
    return {
        k: float(roc_auc_score(y, np.concatenate([uncertainty_value(p, k)[o["counted"]] for p, o in zip(predictions, outcomes)])))
        for k in keys
    }
