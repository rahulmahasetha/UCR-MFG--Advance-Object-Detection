"""Qualitative figure for the paper: single-view UCR-MFG (top row) vs. + TTA (bottom row) on four UAVDT test frames.

Frames are taken from the 3,318 evaluation frames (every 5th test frame from the third), one per condition:
dense daylight traffic, night, fog, high altitude. Within each condition, a few candidate frames with a readable
number of vehicles are scored by single-view F1 (score >= 0.3, IoU >= 0.5) and the best one is shown.

  python scripts/make_qualitative.py --weights models/stage4_ep100_final.pth
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.data.uavdt import UAVDTDataset  # noqa: E402
from src.engine import LocationCache  # noqa: E402
from tune_inference import DEFAULT_SETTING, load_model, merge, pick_device, run_views, view_keys_for  # noqa: E402

CLASS_COLORS = ("#00e676", "#ff9100", "#2979ff")  # car, truck, bus
IMG_SIZE = (544, 1024)
FIELDS = ("boxes", "scores", "labels", "ignored")
CONDITIONS = [  # (column title, predicate on sequence attributes, GT-count range)
    ("Daylight, dense traffic", lambda a: "daylight" in a["weather"] and "high" not in a["altitude"], (45, 90)),
    ("Night", lambda a: "night" in a["weather"], (20, 70)),
    ("Fog", lambda a: "fog" in a["weather"], (15, 70)),
    ("High altitude", lambda a: "high" in a["altitude"], (20, 90)),
]


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def f1(det, gt_boxes, thr):
    keep = (det["scores"] >= thr) & ~det["ignored"]
    boxes = det["boxes"][keep]
    if len(gt_boxes) == 0 or len(boxes) == 0:
        return 0.0
    m = iou(boxes, gt_boxes)
    tp, used = 0, set()
    for i in np.argsort(-det["scores"][keep]):
        j = int(np.argmax(m[i]))
        if m[i, j] >= 0.5 and j not in used:
            used.add(j); tp += 1
    p, r = tp / len(boxes), tp / len(gt_boxes)
    return 2 * p * r / (p + r + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/stage4_ep100_final.pth")
    ap.add_argument("--choice", default="results/final_model/stage4_end_choice.json")
    ap.add_argument("--candidates", type=int, default=8, help="frames scored per condition")
    ap.add_argument("--score-thresh", type=float, default=0.3, help="only draw detections above this score")
    ap.add_argument("--out", default="paper/figures/qualitative.png")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", nargs=4, default=["auto"] * 4,
                    help="image id per condition (e.g. M0403_img000169) or 'auto' to select")
    ap.add_argument("--both", action="store_true",
                    help="select by min(single-view F1, TTA F1) instead of single-view F1")
    ap.add_argument("--sequences", nargs="*", default=None, help="restrict auto-selected frames to these sequences")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, _ = load_model([args.weights], device)
    locations = LocationCache(model.det_head)
    tta = json.load(open(args.choice))["by_fold"]["0"]
    ds = UAVDTDataset("dataset/UAVDT-Benchmark-M", "test", IMG_SIZE, is_train=False, frame_stride=1)
    eval_idx = list(range(2, len(ds), 5))
    rng = random.Random(args.seed)

    def views_for(i, setting):
        s = ds[i]
        with torch.no_grad():
            v = run_views(model, locations, s["image"], setting["scales"], setting["flip"], device, args.seed, i,
                          s["ignore_mask"].numpy(), FIELDS)
        return s, merge(v, view_keys_for(setting), setting, FIELDS)

    by_id = {f"{seq}_img{num:06d}": i for i, (seq, _, num) in enumerate(ds.frames)}
    chosen = []
    for (title, pred, (lo, hi)), fixed in zip(CONDITIONS, args.frames):
        if fixed != "auto":
            chosen.append((title, by_id[fixed]))
            print(f"{title:<26} frame {fixed}  (fixed)")
            continue
        pool = [i for i in eval_idx if pred(ds.sequence_attributes[ds.frames[i][0]])
                and (not args.sequences or ds.frames[i][0] in args.sequences)]
        rng.shuffle(pool)
        scored = []
        for i in pool:
            n_gt = len(ds[i]["boxes"])
            if not lo <= n_gt <= hi:
                continue
            s, det = views_for(i, DEFAULT_SETTING)
            score = f1(det, s["boxes"].numpy(), args.score_thresh)
            if args.both:
                score = min(score, f1(views_for(i, tta)[1], s["boxes"].numpy(), args.score_thresh))
            scored.append((score, i, n_gt))
            if len(scored) >= args.candidates:
                break
        best = max(scored)
        print(f"{title:<26} frame {ds[best[1]]['image_id']}  GT {best[2]}  F1 {best[0]:.2f}  "
              f"(from {len(scored)} candidates)")
        chosen.append((title, best[1]))

    fig, axes = plt.subplots(2, 4, figsize=(7.16, 2.05), dpi=400)
    plt.subplots_adjust(left=0.035, right=1.0, top=0.93, bottom=0.0, wspace=0.015, hspace=0.03)
    for col, (title, i) in enumerate(chosen):
        for row, setting in enumerate((DEFAULT_SETTING, tta)):
            s, det = views_for(i, setting)
            ax = axes[row, col]
            ax.imshow(s["image"].permute(1, 2, 0).numpy()[2:-2])
            ign = s["ignore_mask"].numpy()[2:-2]
            if ign.any():
                ax.imshow(np.ma.masked_where(ign == 0, ign), cmap="Greys", alpha=0.3, vmin=0, vmax=1)
            keep = det["scores"] >= args.score_thresh
            for (x1, y1, x2, y2), c in zip(det["boxes"][keep], det["labels"][keep]):
                ax.add_patch(Rectangle((x1, y1 - 2), x2 - x1, y2 - y1, fill=False, ec=CLASS_COLORS[int(c)], lw=0.45))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3)
            if row == 0:
                ax.set_title(title, fontsize=6, pad=2)
            if col == 0:
                ax.set_ylabel("UCR-MFG" if row == 0 else "+ TTA", fontsize=6, labelpad=2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=400)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
