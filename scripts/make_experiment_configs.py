import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy

import yaml

BASE_EXPERIMENT = "configs/experiments/visdrone_primary.yaml"
BASE_MODEL = "configs/model/default.yaml"

# variant -> (description, model overrides, training overrides)
VARIANTS = {
    "ucr_mfg": ("Full UCR-MFG (C1-C4)", {}, {}),
    "det_only": (
        "Baseline detector: backbone + FCOS head, no graph and no C1 supervision",
        {"model.uncertainty.mc_passes": 1, "loss.lambda_u": 0.0}, {"stages": "stage1_only"},
    ),
    "mfg_style": (
        "MFG-style reimplementation: dense fuzzy graph, ungated messages, additive fusion, no uncertainty",
        {"model.router.enabled": False, "model.router.use_uncertainty": False, "model.mpnn.benefit_gate": False,
         "model.fusion.mode": "add", "model.uncertainty.mc_passes": 1}, {},
    ),
    "no_router": ("w/o C2 routing: every candidate edge passes messages", {"model.router.enabled": False}, {}),
    "router_no_uncertainty": ("C2 without Ua/Ue (scores and budgets)", {"model.router.use_uncertainty": False}, {}),
    "no_adaptive_budget": ("C2 with fixed budgets (k_gain = 0)", {"model.router.k_gain": 0.0}, {}),
    "no_benefit_loss": ("C2 without benefit supervision", {"loss.lambda_benefit": 0.0, "loss.benefit_step_interval": 1000000000}, {}),
    "no_benefit_gate": ("C3 without the benefit gate", {"model.mpnn.benefit_gate": False}, {}),
    "fusion_fixed": ("C4 replaced by fixed 0.5 fusion", {"model.fusion.mode": "fixed"}, {}),
    "fusion_add": ("C4 replaced by additive fusion", {"model.fusion.mode": "add"}, {}),
    "no_epistemic": ("C1 without epistemic uncertainty (single pass)", {"model.uncertainty.mc_passes": 1}, {}),
}

SCHEDULES = {
    "full": {"max_epochs": 150, "stages": [(1, 30), (31, 60), (61, 90), (91, 150)]},
    "short": {"max_epochs": 50, "stages": [(1, 10), (11, 20), (21, 30), (31, 50)]},
}


def set_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def stage_block(schedule: dict, stage1_only: bool) -> dict:
    if stage1_only:
        return {"stage_1": {"epochs": [1, schedule["max_epochs"]]}}
    return {f"stage_{i}": {"epochs": list(r)} for i, r in enumerate(schedule["stages"], 1)}


def main():
    parser = argparse.ArgumentParser(description="Write training configs for the paper's baselines and ablations")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--schedule", choices=list(SCHEDULES), default="full")
    parser.add_argument("--out", default="configs/paper")
    args = parser.parse_args()

    with open(BASE_EXPERIMENT) as f:
        base_exp = yaml.safe_load(f)
    with open(BASE_MODEL) as f:
        base_model = yaml.safe_load(f)
    schedule = SCHEDULES[args.schedule]
    os.makedirs(os.path.join(args.out, "model"), exist_ok=True)

    written = []
    for variant in args.variants:
        description, model_over, train_over = VARIANTS[variant]
        model_cfg = copy.deepcopy(base_model)
        for path, value in model_over.items():
            set_path(model_cfg, path, value)
        model_path = os.path.join(args.out, "model", f"{variant}.yaml")
        with open(model_path, "w") as f:
            yaml.safe_dump(model_cfg, f, sort_keys=False)

        for seed in args.seeds:
            name = f"paper_{variant}_{args.schedule}_s{seed}"
            exp = copy.deepcopy(base_exp)
            exp["experiment_name"] = name
            exp["model_config"] = model_path
            exp["train"]["max_epochs"] = schedule["max_epochs"]
            exp["train"]["seed"] = seed
            exp["train"]["stages"] = stage_block(schedule, train_over.get("stages") == "stage1_only")
            exp["checkpoint"]["save_dir"] = f"checkpoints/paper/{name}"
            exp["paper"] = {"variant": variant, "seed": seed, "schedule": args.schedule, "description": description}
            exp_path = os.path.join(args.out, f"{name}.yaml")
            with open(exp_path, "w") as f:
                yaml.safe_dump(exp, f, sort_keys=False)
            written.append(exp_path)

    print(f"Wrote {len(written)} experiment configs ({args.schedule} schedule) to {args.out}/")
    for path in written:
        print(f"  python scripts/train.py --config {path}")


if __name__ == "__main__":
    main()
