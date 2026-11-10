from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.metrics import detection_outcomes, uncertainty_value

UNCERTAINTY_KEYS = ("u_a", "u_e", "box_std", "1-score")


def _pooled(preds, outcomes, key: str):
    values = np.concatenate([uncertainty_value(p, key)[o["counted"]] for p, o in zip(preds, outcomes)])
    correct = np.concatenate([o["is_tp"][o["counted"]] for o in outcomes])
    return values, correct


def reliability_curve(preds, gts, num_bins: int = 15, outcomes=None) -> Dict[str, List[float]]:
    """Confidence vs. precision per bin (duplicates and background detections count as errors)."""
    outcomes = outcomes or detection_outcomes(preds, gts)
    scores = np.concatenate([p["scores"][o["counted"]] for p, o in zip(preds, outcomes)])
    correct = np.concatenate([o["is_tp"][o["counted"]] for o in outcomes])
    edges = np.linspace(0, 1, num_bins + 1)
    which = np.clip(np.digitize(scores, edges[1:-1]), 0, num_bins - 1)
    out = {"bin_center": [], "confidence": [], "accuracy": [], "count": []}
    for b in range(num_bins):
        m = which == b
        out["bin_center"].append(float((edges[b] + edges[b + 1]) / 2))
        out["count"].append(int(m.sum()))
        out["confidence"].append(float(scores[m].mean()) if m.any() else float("nan"))
        out["accuracy"].append(float(correct[m].mean()) if m.any() else float("nan"))
    return out


def risk_coverage(preds, gts, keys: Sequence[str] = UNCERTAINTY_KEYS, points: int = 50, outcomes=None) -> Dict[str, Any]:
    """Precision of the detections kept when the most uncertain ones are rejected first.

    AURC is the area under the risk (1 - precision) vs. coverage curve; lower is better.
    The oracle curve rejects errors first and bounds every uncertainty from below.
    """
    outcomes = outcomes or detection_outcomes(preds, gts)
    coverage = np.linspace(1.0 / points, 1.0, points)
    result = {"coverage": coverage.tolist(), "curves": {}, "AURC": {}}
    correct = None
    for key in (*keys, "oracle"):
        if key == "oracle":
            values = (~correct).astype(float)
        else:
            values, correct = _pooled(preds, outcomes, key)
        if values.size == 0:
            continue
        order = np.argsort(values, kind="mergesort")
        cum_correct = np.cumsum(correct[order])
        kept = np.maximum((coverage * values.size).astype(int), 1)
        precision = cum_correct[kept - 1] / kept
        result["curves"][key] = precision.tolist()
        result["AURC"][key] = float(np.trapezoid(1 - precision, coverage))
    return result


def uncertainty_by_occlusion(preds, gts, outcomes=None) -> Dict[str, Dict[str, float]]:
    """Mean Ua / Ue of true positives grouped by the occlusion level of the matched ground truth."""
    if not all("occlusion" in gt for gt in gts):
        return {}
    outcomes = outcomes or detection_outcomes(preds, gts)
    groups: Dict[int, Dict[str, list]] = {k: {"u_a": [], "u_e": [], "box_std": []} for k in range(3)}
    for p, g, o in zip(preds, gts, outcomes):
        tp = np.where(o["is_tp"] & (o["gt_index"] >= 0))[0]
        levels = g["occlusion"][o["gt_index"][tp]]
        for key in groups[0]:
            v = uncertainty_value(p, key)[tp]
            for level in range(3):
                groups[level][key].append(v[levels == level])
    out = {}
    for level, name in enumerate(("none", "partial", "heavy")):
        stats = {}
        for key, chunks in groups[level].items():
            v = np.concatenate(chunks) if chunks else np.zeros(0)
            stats[key] = float(v.mean()) if v.size else float("nan")
            stats[f"{key}_sem"] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
        stats["n"] = int(sum(c.size for c in groups[level]["u_a"]))
        out[name] = stats
    return out


def domain_shift(in_domain: List[Dict[str, np.ndarray]], out_domain: List[Dict[str, np.ndarray]],
                 keys: Sequence[str] = ("u_e", "u_a", "1-score"), top_k: Optional[int] = 100) -> Dict[str, Any]:
    """Whether image-level uncertainty separates out-of-domain (e.g. UAVDT) from in-domain images.

    Each image is scored by the mean uncertainty of its `top_k` most confident detections; AUROC > 0.5 means
    the uncertainty is higher on the unseen domain. Detection-level distributions are returned for plotting.
    """
    def image_scores(preds, key):
        out = []
        for p in preds:
            order = np.argsort(-p["scores"])[:top_k] if top_k else slice(None)
            v = uncertainty_value(p, key)[order]
            out.append(float(v.mean()) if v.size else float("nan"))
        return np.array(out)

    result = {"AUROC": {}, "mean_in": {}, "mean_out": {}, "detections": {}}
    for key in keys:
        a, b = image_scores(in_domain, key), image_scores(out_domain, key)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size and b.size:
            y = np.r_[np.zeros(a.size), np.ones(b.size)]
            result["AUROC"][key] = float(roc_auc_score(y, np.r_[a, b]))
            result["mean_in"][key], result["mean_out"][key] = float(a.mean()), float(b.mean())
        result["detections"][key] = {
            "in": np.concatenate([uncertainty_value(p, key) for p in in_domain]).tolist()[:200000],
            "out": np.concatenate([uncertainty_value(p, key) for p in out_domain]).tolist()[:200000],
        }
    return result
