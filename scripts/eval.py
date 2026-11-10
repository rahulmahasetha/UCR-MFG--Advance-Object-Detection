import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.collate import collate_fn
from src.data.dota import DOTADataset
from src.data.uavdt import UAVDTDataset
from src.data.visdrone import VisDroneDataset
from src.engine import evaluate
from src.models.ucr_mfg import UCRMFG

DATASETS = {"VisDrone2019-DET": VisDroneDataset, "UAVDT": UAVDTDataset, "DOTA-v1.5": DOTADataset}
DEFAULT_MODEL_CONFIG = "configs/model/default.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="UCR-MFG evaluation")
    parser.add_argument("--weights", default=None, help="Checkpoint (defaults to the config's weights_checkpoint)")
    parser.add_argument("--config", default="configs/experiments/uavdt_zeroshot.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default=None, help="Optional JSON path for the metrics")
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def label_map_from(exp_cfg: dict, eval_names: list):
    mapping = exp_cfg.get("evaluation", {}).get("ontology_mapping")
    if not mapping:
        return None
    return {src: eval_names.index(name) for name, sources in mapping.items() for src in sources}


def main():
    args = parse_args()
    exp_cfg = load_yaml(args.config)
    ds_cfg = load_yaml(exp_cfg["dataset_config"])
    weights = args.weights or exp_cfg.get("weights_checkpoint")
    split = args.split or exp_cfg.get("evaluation", {}).get("split", "test")
    device = torch.device(args.device)

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    saved = checkpoint.get("config", {})
    model_cfg = saved.get("model") or load_yaml(exp_cfg.get("model_config", DEFAULT_MODEL_CONFIG))
    train_classes = saved.get("dataset", {}).get("classes", {}).get("num_classes", 10)

    model = UCRMFG.from_config(model_cfg, train_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_curriculum_stage(checkpoint.get("current_stage", 4))

    dataset = DATASETS[ds_cfg["dataset_name"]](ds_cfg["root_dir"], split, tuple(ds_cfg["image_size"]), is_train=False)
    if args.max_samples:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    eval_names = ds_cfg["classes"]["names"]
    label_map = label_map_from(exp_cfg, eval_names)
    print(f"[EVAL] {ds_cfg['dataset_name']} ({split}), {len(dataset)} images, stage {model.stage}, weights {weights}")
    metrics = evaluate(model, loader, device, model_cfg["inference"], len(eval_names), label_map=label_map)

    print(f"  mAP@[.5:.95] {metrics['mAP']:.4f} | AP50 {metrics['AP50']:.4f} | AP75 {metrics['AP75']:.4f}")
    print(f"  APs {metrics['APs']:.4f} | APm {metrics['APm']:.4f} | APl {metrics['APl']:.4f}")
    print(f"  D-ECE {metrics['D-ECE']:.4f}")
    print("  Error-detection AUROC: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics["AUROC"].items()))
    for name, ap in zip(eval_names, metrics["per_class_AP"]):
        print(f"    {name:<18} {ap:.4f}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
