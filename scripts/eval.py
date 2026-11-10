import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

import torch

from src.data.corruptions import CORRUPTIONS
from src.evaluation import run_evaluation


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one UCR-MFG checkpoint on one dataset split")
    parser.add_argument("--weights", required=True, help="Checkpoint written by scripts/train.py")
    parser.add_argument("--config", default="configs/experiments/visdrone_primary.yaml",
                        help="Experiment config naming the dataset (and ontology mapping for zero-shot)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--corruption", choices=sorted(CORRUPTIONS), default=None)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--mc-passes", type=int, default=None, help="Override MC dropout passes T at test time (sensitivity study)")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every N-th image (UAVDT video frames)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-predictions", action="store_true", help="Do not save per-detection predictions")
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def main():
    args = parse_args()
    path = run_evaluation(
        args.weights, args.config, args.split, torch.device(args.device),
        corruption=(args.corruption, args.severity) if args.corruption else None,
        stride=args.stride, max_samples=args.max_samples, batch_size=args.batch_size,
        results_dir=args.results_dir, save_predictions=not args.no_predictions, mc_passes=args.mc_passes,
    )
    with open(path) as f:
        result = json.load(f)
    meta, m = result["meta"], result["metrics"]
    print(f"[EVAL] {meta['experiment']} | {meta['dataset']} {meta['split']} | {meta['num_images']} images | "
          f"epoch {meta['epoch']} stage {meta['stage']}" + (f" | {meta['corruption']} s{meta['severity']}" if meta["corruption"] else ""))
    print(f"  mAP {m['mAP']:.4f} | AP50 {m['AP50']:.4f} | AP75 {m['AP75']:.4f} | APs {m['APs']:.4f} | APm {m['APm']:.4f} | APl {m['APl']:.4f}")
    print(f"  D-ECE {m['D-ECE']:.4f} | error AUROC " + ", ".join(f"{k} {v:.3f}" for k, v in m["AUROC"].items()))
    for name, o in m.get("occlusion", {}).items():
        print(f"  occlusion {name:<8} AP50 {o['AP50']:.4f} recall50 {o['recall50']:.4f} (n={o['num_gt']})")
    for name, c in m.get("conditions", {}).items():
        print(f"  {name:<18} mAP {c['mAP']:.4f} AP50 {c['AP50']:.4f} ({c['num_images']} images)")
    for name, ap in zip(meta["class_names"], m["per_class_AP"]):
        print(f"    {name:<18} {ap:.4f}")
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
