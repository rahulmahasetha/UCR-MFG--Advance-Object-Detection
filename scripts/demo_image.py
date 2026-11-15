"""Run UCR-MFG on one UAVDT image and save a picture of its detections with their uncertainties.

Examples:
  python scripts/demo_image.py --weights models/stage4_ep100_final.pth --index 2
  python scripts/demo_image.py --weights models/stage4_ep100_final.pth --image my_frame.jpg

Detections are coloured by class; each label shows score, aleatoric (Ua) and epistemic (Ue) uncertainty.
Ground-truth boxes (test-split images only) are drawn as dashed white outlines; ignore regions are shaded.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.transforms import resize_and_pad  # noqa: E402
from src.data.uavdt import UAVDTDataset  # noqa: E402
from src.engine import LocationCache  # noqa: E402
from src.evaluation import load_trained_model  # noqa: E402
from src.models.postprocess import postprocess  # noqa: E402
from src.models.uncertainty_head import mc_generator  # noqa: E402

CLASS_NAMES = ("car", "truck", "bus")
CLASS_COLORS = ("#2f9e44", "#e8590c", "#1c7ed6")
IMG_SIZE = (544, 1024)  # training canvas (H, W)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/stage4_ep100_final.pth")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--index", type=int, default=2, help="frame index in the UAVDT test split")
    src.add_argument("--image", help="path to any image instead of a test frame")
    ap.add_argument("--score-thresh", type=float, default=0.5, help="only draw detections above this score")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/demo")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, meta = load_trained_model(args.weights, device)
    model.eval()

    gt_boxes, gt_labels, ignore, title = None, None, None, None
    if args.image:
        with Image.open(args.image) as img:
            image, _, _ = resize_and_pad(img.convert("RGB"), IMG_SIZE)
        name = os.path.splitext(os.path.basename(args.image))[0]
        title = os.path.basename(args.image)
    else:
        ds = UAVDTDataset("dataset/UAVDT-Benchmark-M", "test", IMG_SIZE, is_train=False, frame_stride=1)
        sample = ds[args.index]
        image, gt_boxes, gt_labels = sample["image"], sample["boxes"].numpy(), sample["labels"].numpy()
        ignore = sample["ignore_mask"].numpy()
        seq = ds.frames[args.index][0]
        attr = ds.sequence_attributes[seq]
        name = sample["image_id"]
        title = f"{name}  ({', '.join(attr['weather'] + attr['altitude'])} altitude)"

    t0 = time.time()
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)
        outputs = model(x, generator=mc_generator(0, 0, 0, device))
        locs = LocationCache(model.det_head)(outputs["head"]["cls"])
        det = postprocess(outputs, locs, model.strides, x.shape[-2:], **meta["inference"])[0]
    elapsed = time.time() - t0
    det = {k: v.cpu().numpy() for k, v in det.items()}
    keep = det["scores"] >= args.score_thresh
    det = {k: v[keep] for k, v in det.items()}

    fig, ax = plt.subplots(figsize=(16, 8.8), dpi=110)
    ax.imshow(image.permute(1, 2, 0).numpy())
    if ignore is not None and ignore.any():
        ax.imshow(np.ma.masked_where(ignore == 0, ignore), cmap="Greys", alpha=0.35, vmin=0, vmax=1)
    if gt_boxes is not None:
        for b in gt_boxes:
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="white", lw=0.9, ls="--"))
    order = np.argsort(det["scores"])
    for i in order:
        x1, y1, x2, y2 = det["boxes"][i]
        c = int(det["labels"][i])
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=CLASS_COLORS[c], lw=1.6))
        ax.text(x1, y1 - 2, f"{CLASS_NAMES[c]} {det['scores'][i]:.2f}\nUa {det['u_a'][i]:.2f} Ue {det['u_e'][i]:.2f}",
                fontsize=5.5, color="white", va="bottom",
                bbox=dict(facecolor=CLASS_COLORS[c], alpha=0.75, pad=0.6, lw=0))
    counts = ", ".join(f"{int((det['labels'] == c).sum())} {n}" for c, n in enumerate(CLASS_NAMES))
    gt_txt = f" | ground truth: {len(gt_boxes)} vehicles (dashed)" if gt_boxes is not None else ""
    ax.set_title(f"{title}\nUCR-MFG epoch {meta['epoch']}: {len(det['scores'])} detections above {args.score_thresh} "
                 f"({counts}){gt_txt} | {elapsed:.1f} s on {device}", fontsize=10)
    ax.axis("off")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"{len(det['scores'])} detections above {args.score_thresh} ({counts}); {elapsed:.1f} s on {device}")
    if gt_boxes is not None:
        print(f"ground truth: {len(gt_boxes)} vehicles")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
