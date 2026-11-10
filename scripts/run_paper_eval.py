import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

import torch

from src.data.corruptions import CORRUPTIONS
from src.evaluation import UnusableCheckpointError, load_trained_model, run_evaluation

VISDRONE = "configs/experiments/visdrone_primary.yaml"
UAVDT = "configs/experiments/uavdt_zeroshot.yaml"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Run every evaluation the paper needs for one trained checkpoint")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--suite", nargs="+", default=["val", "test_dev", "uavdt", "corruptions"],
                        choices=["val", "test_dev", "uavdt", "corruptions"])
    parser.add_argument("--uavdt-stride", type=int, default=10, help="Every N-th UAVDT frame (1 = all 16,592)")
    parser.add_argument("--corruptions", nargs="+", default=sorted(CORRUPTIONS), choices=sorted(CORRUPTIONS))
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--force", action="store_true", help="Re-run evaluations whose result file already exists")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    device = torch.device(args.device)

    try:
        _, meta = load_trained_model(args.weights, torch.device("cpu"))
    except UnusableCheckpointError as err:
        sys.exit(f"[ERROR] {err}")
    if meta["mid_epoch"] or (meta["max_epochs"] and meta["epoch"] < meta["max_epochs"]):
        print(f"[WARN] {args.weights} is from epoch {meta['epoch']}/{meta['max_epochs']} (stage {meta['stage']}); "
              "results will be flagged as incomplete training.")

    jobs = []
    if "val" in args.suite:
        jobs.append(("visdrone_val", VISDRONE, "val", None, 1))
    if "test_dev" in args.suite:
        jobs.append(("visdrone_test_dev", VISDRONE, "test_dev", None, 1))
    if "uavdt" in args.suite:
        jobs.append(("uavdt_test", UAVDT, "test", None, args.uavdt_stride))
    if "corruptions" in args.suite:
        jobs += [(f"visdrone_val_{c}{s}", VISDRONE, "val", (c, s), 1) for c in args.corruptions for s in args.severities]

    out_dir = os.path.join(args.results_dir, meta["experiment"])
    for n, (tag, cfg, split, corruption, stride) in enumerate(jobs, 1):
        target = os.path.join(out_dir, f"{tag}.json")
        if os.path.exists(target) and not args.force:
            with open(target) as f:
                if json.load(f)["meta"]["weights"] == meta["weights"]:
                    print(f"[{n}/{len(jobs)}] {tag}: exists, skipping")
                    continue
        print(f"[{n}/{len(jobs)}] {tag} ...", flush=True)
        path = run_evaluation(args.weights, cfg, split, device, corruption=corruption, stride=stride,
                              results_dir=args.results_dir, save_predictions=corruption is None)
        with open(path) as f:
            m = json.load(f)["metrics"]
        print(f"    mAP {m['mAP']:.4f} AP50 {m['AP50']:.4f} -> {path}")


if __name__ == "__main__":
    main()
