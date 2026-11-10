import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.collate import collate_fn
from src.data.visdrone import VisDroneDataset
from src.losses.router_loss import same_instance_edges
from src.models.graph_ops import NUM_EDGE_TYPES
from src.models.uncertainty_head import mc_generator
from src.models.ucr_mfg import UCRMFG

EDGE_NAMES = ("local", "knn", "top-down", "bottom-up")


def main():
    parser = argparse.ArgumentParser(description="Router diagnostic against a GT same-object oracle")
    parser.add_argument("--config", default="configs/experiments/visdrone_primary.yaml")
    parser.add_argument("--weights", default=None, help="Trained checkpoint; without it the router is untrained")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        exp_cfg = yaml.safe_load(f)
    with open(exp_cfg["dataset_config"]) as f:
        ds_cfg = yaml.safe_load(f)
    with open(exp_cfg["model_config"]) as f:
        model_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = UCRMFG.from_config(model_cfg, ds_cfg["classes"]["num_classes"]).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=False)["model_state_dict"])
    model.set_curriculum_stage(4)
    model.eval()

    dataset = VisDroneDataset(ds_cfg["root_dir"], "val", tuple(ds_cfg["image_size"]))
    loader = DataLoader(Subset(dataset, range(min(args.samples, len(dataset)))), batch_size=1, collate_fn=collate_fn)

    stats = torch.zeros(NUM_EDGE_TYPES, 5, dtype=torch.long)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            outputs = model(batch["images"].to(device), generator=mc_generator(0, 0, i, device))
            graph, routing = outputs["graph"], outputs["routing"]
            relevant = same_instance_edges(graph, batch["boxes"])
            for t in range(NUM_EDGE_TYPES):
                of_type = graph.edge_types == t
                sel = routing.selected & of_type
                n_rel, n_sel = int((relevant & of_type).sum()), int(sel.sum())
                stats[t] += torch.tensor([int(of_type.sum()), n_rel, n_sel, int((relevant & sel).sum()), min(n_rel, n_sel)])

    print(f"\nRouter vs. same-object oracle on {len(loader)} VisDrone val images ({'trained' if args.weights else 'UNTRAINED'} router)")
    print(f"{'edge type':<10} {'candidates':>11} {'base rate':>10} {'router prec':>12} {'oracle prec':>12} {'router recall':>14}")
    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    for t, name in enumerate(EDGE_NAMES):
        cand, rel, sel, rel_sel, oracle = stats[t].tolist()
        print(f"{name:<10} {cand:>11} {ratio(rel, cand):>10.4f} {ratio(rel_sel, sel):>12.4f} {ratio(oracle, sel):>12.4f} {ratio(rel_sel, rel):>14.4f}")
    print("Oracle precision is the best achievable with the same number of selected edges per image.\n")


if __name__ == "__main__":
    main()
