import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import datetime
import glob
import json
import time

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from src.evaluation import load_trained_model
from src.models.ucr_mfg import UCRMFG


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def parameter_counts(model: UCRMFG, stage: int) -> dict:
    used = ["backbone", "unc_head", "det_head"] + (["graph_builder"] if stage >= 2 else []) + \
           (["router", "mpnn"] if stage >= 3 else []) + (["fusion_gate"] if stage >= 4 else [])
    per_module = {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in used}
    return {"total_M": sum(per_module.values()) / 1e6, "per_module_M": {k: v / 1e6 for k, v in per_module.items()}}


@torch.no_grad()
def measure(model: UCRMFG, image_size, device: torch.device, iters: int, warmup: int) -> dict:
    model.eval().to(device)
    images = torch.rand(1, 3, *image_size, device=device)
    for _ in range(warmup):
        model(images)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = model(images)
        sync(device)
        times.append(time.perf_counter() - start)
    times.sort()
    median = times[len(times) // 2]
    result = {"latency_ms": 1000 * median, "fps": 1.0 / median, "latency_ms_p90": 1000 * times[int(0.9 * (len(times) - 1))]}
    if device.type == "cuda":
        result["peak_memory_GB"] = torch.cuda.max_memory_allocated() / 2 ** 30
    elif device.type == "mps":
        result["driver_memory_GB"] = torch.mps.driver_allocated_memory() / 2 ** 30
    if out["graph"] is not None:
        result["candidate_edges"] = int(out["graph"].edge_index.shape[1])
        result["routed_edges"] = int(out["routing"].selected.sum()) if out["routing"] is not None else result["candidate_edges"]
        result["routed_fraction"] = result["routed_edges"] / max(result["candidate_edges"], 1)
    return result


@torch.no_grad()
def gflops(model: UCRMFG, image_size) -> float:
    model.eval().cpu()
    counter = FlopCounterMode(display=False)
    with counter:
        model(torch.rand(1, 3, *image_size))
    return counter.get_total_flops() / 1e9


def main():
    parser = argparse.ArgumentParser(description="Parameters, GFLOPs, latency and graph sparsity per model variant")
    parser.add_argument("--model-configs", nargs="*", default=None, help="Model YAMLs (default: configs/paper/model/*.yaml)")
    parser.add_argument("--weights", nargs="*", default=[], help="Trained checkpoints (routing statistics then reflect trained routers)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--out", default="results/efficiency")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    models = []
    for path in args.weights:
        model, meta = load_trained_model(path, torch.device("cpu"))
        models.append((meta["variant"], model, model.stage, {"weights": meta["weights"], "trained": True}))
    configs = args.model_configs if args.model_configs is not None else (sorted(glob.glob("configs/paper/model/*.yaml")) or ["configs/model/default.yaml"])
    if not args.weights:
        for path in configs:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            cfg = copy.deepcopy(cfg)
            cfg["model"]["pretrained"] = False
            name = os.path.splitext(os.path.basename(path))[0]
            stage = 1 if name == "det_only" else 4
            model = UCRMFG.from_config(cfg, args.num_classes)
            model.set_curriculum_stage(stage)
            models.append((name, model, stage, {"model_config": path, "trained": False}))

    for name, model, stage, source in models:
        print(f"[BENCH] {name} (stage {stage}) ...", flush=True)
        result = {"variant": name, "stage": stage, **source, **parameter_counts(model, stage)}
        mc_passes = model.unc_head.mc_passes
        result["mc_passes"] = mc_passes
        if not args.skip_flops:
            result["GFLOPs"] = gflops(model, args.image_size)
            model.unc_head.mc_passes = 1
            result["GFLOPs_single_pass"] = gflops(model, args.image_size)
            model.unc_head.mc_passes = mc_passes
        result.update(measure(model, args.image_size, device, args.iters, args.warmup))
        result["meta"] = {
            "device": str(device), "image_size": args.image_size, "batch_size": 1, "precision": "fp32",
            "torch": torch.__version__, "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "note": "GFLOPs count convolutions and matrix products; graph gather/scatter is memory-bound and not counted.",
        }
        if not source["trained"]:
            result["meta"]["note"] += " Untrained weights: routed-edge counts reflect an untrained router."
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(result, f, indent=2)
        line = f"    params {result['total_M']:.1f}M | latency {result['latency_ms']:.0f} ms ({result['fps']:.2f} FPS)"
        if "GFLOPs" in result:
            line += f" | {result['GFLOPs']:.0f} GFLOPs ({result['GFLOPs_single_pass']:.0f} with 1 MC pass)"
        if "candidate_edges" in result:
            line += f" | routed {result['routed_edges']}/{result['candidate_edges']}"
        print(line)
        model.cpu()


if __name__ == "__main__":
    main()
