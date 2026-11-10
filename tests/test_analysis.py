import numpy as np

from src.analysis.uncertainty import domain_shift, reliability_curve, risk_coverage, uncertainty_by_occlusion
from src.utils.metrics import detection_outcomes, uncertainty_auroc


def synthetic(n_images=20, seed=0):
    """Each image: 3 GT boxes (occlusion 0/1/2) detected exactly, plus 2 false positives with high uncertainty."""
    rng = np.random.default_rng(seed)
    preds, gts = [], []
    for _ in range(n_images):
        gt_boxes = np.array([[0, 0, 40, 40], [100, 100, 150, 160], [300, 300, 340, 330]], float)
        fp_boxes = np.array([[600, 600, 640, 640], [800, 50, 850, 90]], float)
        occl = np.array([0, 1, 2])
        u_a_tp = np.array([0.1, 0.3, 0.6])
        preds.append({
            "boxes": np.vstack([gt_boxes, fp_boxes]), "labels": np.array([0, 0, 0, 0, 0]),
            "scores": np.array([0.9, 0.8, 0.7, 0.85, 0.75]),
            "u_a": np.r_[u_a_tp, 0.8, 0.9], "u_e": np.r_[rng.uniform(0.0, 0.2, 3), 0.7, 0.9],
            "box_std": np.tile(np.r_[1.0, 2.0, 3.0, 9.0, 9.0][:, None], (1, 4)),
        })
        gts.append({"boxes": gt_boxes, "labels": np.zeros(3, int), "occlusion": occl})
    return preds, gts


def test_auroc_and_risk_coverage_rank_errors_last():
    preds, gts = synthetic()
    outcomes = detection_outcomes(preds, gts)
    assert sum(o["is_tp"].sum() for o in outcomes) == 60
    auroc = uncertainty_auroc(preds, gts, outcomes=outcomes)
    assert auroc["u_e"] == 1.0 and auroc["box_std"] == 1.0
    rc = risk_coverage(preds, gts, points=10, outcomes=outcomes)
    assert np.isclose(rc["curves"]["u_e"][5], 1.0), "keeping the 60% most certain detections keeps only true positives"
    assert np.isclose(rc["AURC"]["u_e"], rc["AURC"]["oracle"])
    assert rc["AURC"]["1-score"] > rc["AURC"]["u_e"], "confidence alone ranks these errors worse"


def test_uncertainty_by_occlusion_and_reliability():
    preds, gts = synthetic()
    occ = uncertainty_by_occlusion(preds, gts)
    assert [round(occ[k]["u_a"], 3) for k in ("none", "partial", "heavy")] == [0.1, 0.3, 0.6]
    assert all(occ[k]["n"] == 20 for k in occ)
    rel = reliability_curve(preds, gts, num_bins=10)
    assert sum(rel["count"]) == 100


def test_domain_shift_detects_higher_uncertainty_out_of_domain():
    preds_in, _ = synthetic(seed=1)
    preds_out, _ = synthetic(seed=2)
    for p in preds_out:
        p["u_e"] = p["u_e"] + 0.5
    result = domain_shift(preds_in, preds_out, keys=("u_e",))
    assert result["AUROC"]["u_e"] == 1.0
    assert result["mean_out"]["u_e"] > result["mean_in"]["u_e"]
