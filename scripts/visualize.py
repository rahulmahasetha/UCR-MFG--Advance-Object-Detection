import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from src.data.collate import collate_fn
from src.data.visdrone import VisDroneDataset
from src.engine import LocationCache
from src.evaluation import load_trained_model
from src.models.graph_ops import NUM_EDGE_TYPES
from src.models.postprocess import postprocess
from src.models.uncertainty_head import mc_generator

EDGE_COLORS = ("#3b2f7a", "#2f7a4a", "#c07a1e", "#b4432f")
EDGE_NAMES = ("local", "kNN", "top-down", "bottom-up")


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def occluded_first(dataset: VisDroneDataset, count: int):
    """Indices of the images with the most heavily occluded annotated objects."""
    heavy = []
    for i, (_, anno) in enumerate(dataset.samples):
        n = 0
        if os.path.exists(anno):
            with open(anno) as f:
                for line in f:
                    p = line.strip().split(",")
                    n += len(p) >= 8 and p[5] not in ("0", "11") and p[4] != "0" and p[7] == "2"
        heavy.append(n)
    return list(np.argsort(heavy)[::-1][:count])


@torch.no_grad()
def render(model, sample, device, out_path, score_thresh, num_routed):
    batch = collate_fn([sample])
    image = batch["images"].to(device)
    gates = []
    hook = model.fusion_gate.gate_conv.register_forward_hook(lambda m, i, o: gates.append(torch.sigmoid(o).mean(1)[0].cpu()))
    outputs = model(image, generator=mc_generator(0, 0, 0, device))
    hook.remove()
    locs = LocationCache(model.det_head)(outputs["head"]["cls"])
    det = {k: v.cpu().numpy() for k, v in postprocess(outputs, locs, model.strides, image.shape[-2:], score_thresh=score_thresh)[0].items()}

    rgb = sample["image"].permute(1, 2, 0).numpy()
    h, w = rgb.shape[:2]
    ua, ue = outputs["u_a"][0][0, 0].cpu().numpy(), outputs["u_e"][0][0, 0].cpu().numpy()
    panels = [("Detections (colour = epistemic Ue)", None), ("Aleatoric Ua (fine level)", ua), ("Epistemic Ue (fine level)", ue)]
    if outputs["routing"] is not None:
        panels.append(("Routed links of the most uncertain detections", None))
    if gates:
        panels.append(("C4 fusion weight λ (1 = visual F)", gates[0].numpy()))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (title, heat) in zip(axes, panels):
        ax.imshow(rgb)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if heat is not None:
            im = ax.imshow(heat, extent=(0, w, h, 0), cmap="magma", alpha=0.55, vmin=0, vmax=max(float(heat.max()), 1e-6))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    cmap = plt.get_cmap("viridis")
    ue_det = det["u_e"]
    scale = max(float(ue_det.max()), 1e-6) if ue_det.size else 1.0
    for box, u in zip(det["boxes"], ue_det):
        axes[0].add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, lw=0.9, ec=cmap(u / scale)))
    axes[0].text(4, h - 8, f"{len(det['scores'])} detections ≥ {score_thresh}", color="white", fontsize=7,
                 bbox=dict(fc="black", alpha=0.5, lw=0))

    if outputs["routing"] is not None and ue_det.size:
        ax = axes[3]
        graph, routing = outputs["graph"], outputs["routing"]
        h1, w1 = graph.shapes[0]
        stride = model.strides[0]
        recv, send = graph.edge_index.cpu().numpy()
        selected = routing.selected.cpu().numpy()
        types = graph.edge_types.cpu().numpy()
        pos = graph.positions.cpu().numpy()
        focus = []
        for rank, d in enumerate(np.argsort(-ue_det)[:num_routed]):
            box = det["boxes"][d]
            cx = int(np.clip((box[0] + box[2]) / 2 // stride, 0, w1 - 1))
            cy = int(np.clip((box[1] + box[3]) / 2 // stride, 0, h1 - 1))
            node = cy * w1 + cx
            for t in range(NUM_EDGE_TYPES):
                m = (recv == node) & selected & (types == t)
                if m.any():
                    segs = [[pos[node], pos[s]] for s in send[m]]
                    ax.add_collection(LineCollection(segs, colors=EDGE_COLORS[t], linewidths=2.0))
                    ax.plot(pos[send[m], 0], pos[send[m], 1], "o", ms=2.5, color=EDGE_COLORS[t])
                    if rank == 0:
                        focus.extend(pos[send[m]])
            ax.plot(*pos[node], "o", ms=5, color="white", mec="black", mew=0.8)
            if rank == 0:
                focus.append(pos[node])
        if focus:
            pts = np.array(focus)
            (x0, y0), (x1, y1) = pts.min(0) - 48, pts.max(0) + 48
            side = max(x1 - x0, y1 - y0, 160)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ax.set_xlim(cx - side / 2, cx + side / 2)
            ax.set_ylim(cy + side / 2, cy - side / 2)
            ax.set_title("Routed links of the most uncertain detection (zoomed)", fontsize=9)
        for t in range(NUM_EDGE_TYPES):
            ax.plot([], [], color=EDGE_COLORS[t], label=EDGE_NAMES[t])
        ax.legend(loc="lower right", fontsize=6, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Qualitative figures: detections, Ua/Ue maps, routed links, fusion weights")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--num", type=int, default=6, help="Number of heavily occluded images when --indices is omitted")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--num-routed", type=int, default=3)
    parser.add_argument("--out", default="paper/figures/qualitative")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    device = torch.device(args.device)

    model, meta = load_trained_model(args.weights, device)
    model.eval()
    dataset = VisDroneDataset("dataset/VisDrone2019-DET", args.split, (1024, 1024), is_train=False)
    indices = args.indices if args.indices else occluded_first(dataset, args.num)
    os.makedirs(args.out, exist_ok=True)
    for i in indices:
        sample = dataset[int(i)]
        path = os.path.join(args.out, f"{meta['experiment']}_{sample['image_id']}.png")
        render(model, sample, device, path, args.score_thresh, args.num_routed)
        print(f"[VIS] {path}")


if __name__ == "__main__":
    main()
