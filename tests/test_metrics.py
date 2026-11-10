import numpy as np

from src.losses.router_loss import benefit_targets
from src.utils.metrics import compute_map, uncertainty_auroc

import torch


def gt(boxes, labels, mask=None):
    return {"boxes": np.array(boxes, float), "labels": np.array(labels), "ignore_mask": mask}


def pred(boxes, scores, labels, u=None):
    n = len(scores)
    u = np.zeros(n) if u is None else np.array(u, float)
    return {
        "boxes": np.array(boxes, float).reshape(-1, 4), "scores": np.array(scores, float), "labels": np.array(labels),
        "u_a": u, "u_e": u, "box_std": np.tile(u[:, None], (1, 4)),
    }


def test_perfect_predictions_give_full_map():
    boxes = [[0, 0, 50, 50], [100, 100, 200, 220]]
    metrics = compute_map([pred(boxes, [0.9, 0.8], [0, 1])], [gt(boxes, [0, 1])], num_classes=2)
    assert np.isclose(metrics["mAP"], 1.0) and np.isclose(metrics["AP50"], 1.0)


def test_false_positive_lowers_map_but_ignore_region_does_not():
    boxes = [[0, 0, 50, 50]]
    fp = [[300, 300, 350, 350]]
    plain = compute_map([pred(boxes + fp, [0.5, 0.9], [0, 0])], [gt(boxes, [0])], 1)
    mask = np.zeros((400, 400), np.uint8)
    mask[290:360, 290:360] = 1
    ignored = compute_map([pred(boxes + fp, [0.5, 0.9], [0, 0])], [gt(boxes, [0], mask)], 1)
    assert plain["mAP"] < 1.0
    assert np.isclose(ignored["mAP"], 1.0)


def test_uncertainty_auroc_detects_errors():
    boxes = [[0, 0, 50, 50]]
    p = pred(boxes + [[300, 300, 350, 350]], [0.9, 0.8], [0, 0], u=[0.1, 0.9])
    auroc = uncertainty_auroc([p], [gt(boxes, [0])])
    assert auroc["u_a"] == 1.0


def test_benefit_targets_select_top_positive_quantile():
    benefit = torch.tensor([-1.0, 0.0, 0.1, 0.2, 0.3, 0.4])
    target = benefit_targets(benefit, 0.75)
    assert target.tolist() == [0, 0, 0, 0, 0, 1]
