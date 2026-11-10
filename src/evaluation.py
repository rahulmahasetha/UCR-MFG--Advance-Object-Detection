import datetime
import hashlib
import json
import os
import pickle
import subprocess
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.collate import collate_fn
from src.data.dota import DOTADataset
from src.data.uavdt import UAVDTDataset
from src.data.visdrone import VisDroneDataset
from src.engine import predict, summarize
from src.models.ucr_mfg import UCRMFG

DATASETS = {"VisDrone2019-DET": VisDroneDataset, "UAVDT": UAVDTDataset, "DOTA-v1.5": DOTADataset}
DEFAULT_INFERENCE = {"score_thresh": 0.05, "pre_nms_topk": 1000, "nms_iou": 0.6, "max_dets": 500}
SHORT_NAMES = {"VisDrone2019-DET": "visdrone", "UAVDT": "uavdt", "DOTA-v1.5": "dota"}


class UnusableCheckpointError(RuntimeError):
    pass


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_trained_model(weights: str, device: torch.device) -> Tuple[UCRMFG, Dict[str, Any]]:
    """Rebuilds the model from the configuration stored in the checkpoint and returns provenance metadata."""
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if not cfg or "model" not in cfg:
        raise UnusableCheckpointError(
            f"{weights} has no embedded configuration; it predates the current training code and cannot be "
            "evaluated reliably. Retrain with scripts/train.py."
        )
    num_classes = cfg["dataset"]["classes"]["num_classes"]
    model = UCRMFG.from_config(cfg["model"], num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.set_curriculum_stage(ckpt.get("current_stage", 4))
    exp = cfg["experiment"]
    paper = exp.get("paper", {})
    meta = {
        "weights": os.path.abspath(weights),
        "experiment": exp.get("experiment_name", "unnamed"),
        "variant": paper.get("variant", exp.get("experiment_name", "unnamed")),
        "seed": exp.get("train", {}).get("seed", 42),
        "epoch": ckpt.get("epoch"),
        "max_epochs": exp.get("train", {}).get("max_epochs"),
        "stage": ckpt.get("current_stage"),
        "mid_epoch": bool(ckpt.get("resume")),
        "best_val_mAP": ckpt.get("best_metric"),
        "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12],
        "train_classes": num_classes,
        "inference": cfg["model"].get("inference", DEFAULT_INFERENCE),
    }
    return model, meta


def run_evaluation(
    weights: str,
    exp_config: str,
    split: str,
    device: torch.device,
    corruption: Optional[Tuple[str, int]] = None,
    stride: int = 1,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    results_dir: str = "results",
    save_predictions: bool = True,
    seed: int = 0,
    mc_passes: Optional[int] = None,
) -> str:
    """Evaluates one checkpoint on one dataset split and writes results/<experiment>/<name>.json (+ .pkl)."""
    model, meta = load_trained_model(weights, device)
    trained_passes = model.unc_head.mc_passes
    if mc_passes is not None:
        model.unc_head.mc_passes = mc_passes
    exp_cfg = load_yaml(exp_config)
    ds_cfg = load_yaml(exp_cfg["dataset_config"])
    ds_name = ds_cfg["dataset_name"]
    names = ds_cfg["classes"]["names"]
    mapping = exp_cfg.get("evaluation", {}).get("ontology_mapping")
    label_map = {src: names.index(n) for n, srcs in mapping.items() for src in srcs} if mapping else None
    if label_map is None and len(names) != meta["train_classes"]:
        raise ValueError(f"{ds_name} has {len(names)} classes but the model was trained on {meta['train_classes']}; add an ontology mapping")

    dataset = DATASETS[ds_name](ds_cfg["root_dir"], split, tuple(ds_cfg["image_size"]), is_train=False)
    indices = list(range(0, len(dataset), stride))
    if max_samples:
        indices = indices[:max_samples]
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=2, persistent_workers=False)

    infer_cfg = meta["inference"]
    preds, gts = predict(model, loader, device, infer_cfg, label_map=label_map, seed=seed, corruption=corruption)
    metrics = summarize(preds, gts, len(names))

    tag = f"{SHORT_NAMES.get(ds_name, ds_name)}_{split}"
    if corruption:
        tag += f"_{corruption[0]}{corruption[1]}"
    if mc_passes is not None and mc_passes != trained_passes:
        tag += f"_T{mc_passes}"
    meta.update({
        "dataset": ds_name, "split": split, "tag": tag, "class_names": names,
        "corruption": corruption[0] if corruption else None, "severity": corruption[1] if corruption else 0,
        "mc_passes": model.unc_head.mc_passes, "trained_mc_passes": trained_passes,
        "stride": stride, "max_samples": max_samples, "num_images": len(preds), "dataset_size": len(dataset),
        "zero_shot_mapping": bool(label_map), "git": git_revision(),
        "device": str(device), "date": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    out_dir = os.path.join(results_dir, meta["experiment"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.json")
    with open(path, "w") as f:
        json.dump({"meta": meta, "metrics": metrics}, f, indent=2, default=float)
    if save_predictions:
        for g in gts:
            g.pop("ignore_mask", None)
        with open(os.path.join(out_dir, f"{tag}.pkl"), "wb") as f:
            pickle.dump({"meta": meta, "predictions": preds, "ground_truths": gts}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_predictions(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
