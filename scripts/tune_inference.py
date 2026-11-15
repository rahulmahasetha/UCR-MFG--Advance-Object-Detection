"""Post-training inference tuning for a trained UCR-MFG checkpoint (no retraining).

Searches test-time settings that do not touch the weights:
  * test-time augmentation (TTA): extra input scales (1.25x, 1.5x) and horizontal flip,
    candidates from all views are mapped back to the original frame and merged;
  * merging / NMS: hard NMS IoU, Matrix-NMS (Gaussian soft decay, Wang et al. 2020);
  * score threshold;
  * optional checkpoint weight averaging (pass several --weights).

Protocol (2-fold cross-tuning by video, keeps the reported test number unbiased):
  The 20 UAVDT test videos are split into two folds A/B (alternating in sorted order). Settings are
  scored on a sparse subset of test frames (default every 30th); the setting used for fold A videos is
  the one chosen on fold B and vice versa, so no image is ever evaluated with settings picked on its
  own video. (Unused frames of the training videos are no use for this: they are 0.5 s away from
  training frames and the model scores mAP ~0.7 on them, so TTA/NMS choices there do not transfer.)
  1. phase "tune": forward passes for every view on the sparse frames, grid of settings, per-fold
     choice; prints the cross-validated mAP on those frames.
  2. phase "test": runs the full test split (stride 1) once; every image gets the setting chosen on
     the other fold. The default setting (1 view, NMS 0.6, thresh 0.05) comes from the same passes.

Usage:
  python scripts/tune_inference.py tune --weights models/stage4_ep100_final.pth
  python scripts/tune_inference.py test --weights models/stage4_ep100_final.pth \
      --choice results/final_model/stage4_end_choice.json
"""
import argparse
import itertools
import json
import os
import pickle
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.ops import batched_nms, box_iou

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.collate import collate_fn  # noqa: E402
from src.data.uavdt import UAVDTDataset  # noqa: E402
from src.engine import LocationCache, summarize  # noqa: E402
from src.evaluation import DEFAULT_INFERENCE, git_revision, load_trained_model, load_yaml  # noqa: E402
from src.models.postprocess import postprocess  # noqa: E402
from src.models.uncertainty_head import mc_generator  # noqa: E402
from src.utils.metrics import _center_ignored, compute_map  # noqa: E402

PAD_VALUE = 114.0 / 255.0     # same grey as resize_and_pad
RAW_THRESH = 0.02             # lowest score threshold searched; candidates below it are never needed
FIELDS_TUNE = ("boxes", "scores", "labels", "ignored")
FIELDS_FULL = ("boxes", "scores", "labels", "ignored", "box_std", "u_a", "u_e")


# --------------------------------------------------------------------------- model / data

def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights: Sequence[str], device: torch.device):
    """Loads one checkpoint, or the element-wise average of several (same architecture)."""
    model, meta = load_trained_model(weights[0], torch.device("cpu"))
    if len(weights) > 1:
        states = [torch.load(w, map_location="cpu", weights_only=False)["model_state_dict"] for w in weights]
        avg = OrderedDict()
        for k, v in states[0].items():
            avg[k] = torch.stack([s[k].float() for s in states]).mean(0).to(v.dtype) if v.is_floating_point() else v
        model.load_state_dict(avg)
        meta["weights"] = [os.path.abspath(w) for w in weights]
        meta["weight_average"] = len(weights)
    model.to(device).eval()
    return model, meta


def build_split(exp_cfg: dict, img_size: Tuple[int, int], stride: int, max_images: int, offset: int = 0):
    """Test split, every `stride`-th frame, plus a 2-fold assignment of its videos."""
    ds_cfg = load_yaml(exp_cfg["dataset_config"])
    if ds_cfg["dataset_name"] != "UAVDT":
        raise SystemExit("tune_inference.py currently supports UAVDT only")
    ds = UAVDTDataset(ds_cfg["root_dir"], "test", img_size, is_train=False, frame_stride=1)
    indices = list(range(offset, len(ds), stride))
    if max_images:
        indices = indices[:max_images]
    seqs = sorted({seq for seq, _, _ in ds.frames})
    folds = {seq: k % 2 for k, seq in enumerate(seqs)}
    return ds, indices, folds, ds_cfg["classes"]["num_classes"], ds_cfg["classes"]["names"]


# --------------------------------------------------------------------------- inference views

def pad_to(x: torch.Tensor, multiple: int = 32) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    return F.pad(x, (0, pw, 0, ph), value=PAD_VALUE) if (ph or pw) else x


def free_device_memory(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def run_views(model, locations, image: torch.Tensor, scales: Sequence[float], flip: bool,
              device: torch.device, seed: int, index: int, ignore_mask: np.ndarray,
              fields: Sequence[str]) -> Dict[Tuple[float, bool], Dict[str, np.ndarray]]:
    """Raw (pre-NMS) candidates of every view, in the coordinates of the unscaled, unflipped frame.

    Views run one at a time (batch 1) and device memory is released after each, so crowded frames at
    1.5x fit in the 16 GB of unified memory.
    """
    h0, w0 = image.shape[-2:]
    out = {}
    for s in scales:
        scaled = image.unsqueeze(0)
        if s != 1.0:
            scaled = F.interpolate(scaled, scale_factor=s, mode="bilinear", align_corners=False)
        sw = scaled.shape[-1]                              # scaled width before padding (for un-flip)
        for is_flip in ((False, True) if flip else (False,)):
            x = pad_to(scaled.flip(-1) if is_flip else scaled).to(device)
            outputs = model(x, generator=mc_generator(seed, int(is_flip), index, device))
            det = postprocess(outputs, locations(outputs["head"]["cls"]), model.strides, x.shape[-2:],
                              score_thresh=RAW_THRESH, pre_nms_topk=DEFAULT_INFERENCE["pre_nms_topk"],
                              nms_iou=1.0, max_dets=10 ** 7)[0]   # IoU 1.0 = no suppression
            det = {k: v.float().cpu() if v.is_floating_point() else v.cpu() for k, v in det.items()}
            del outputs, x
            free_device_memory(device)
            boxes = det["boxes"]
            if is_flip:
                boxes = torch.stack([sw - boxes[:, 2], boxes[:, 1], sw - boxes[:, 0], boxes[:, 3]], 1)
            boxes = boxes / s
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w0)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h0)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            view = {"boxes": boxes[keep].numpy(), "scores": det["scores"][keep].numpy(),
                    "labels": det["labels"][keep].numpy().astype(np.int64)}
            view["ignored"] = _center_ignored(view["boxes"], ignore_mask)
            if "box_std" in fields:
                view["box_std"] = (det["box_std"][keep] / s).numpy()
                view["u_a"] = det["u_a"][keep].numpy()
                view["u_e"] = det["u_e"][keep].numpy()
            out[(s, is_flip)] = view
    return out


# --------------------------------------------------------------------------- merging

def matrix_nms(boxes, scores, labels, sigma: float, max_cands: int = 3000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gaussian Matrix-NMS (SOLOv2): decays scores of overlapping same-class boxes instead of removing them."""
    order = scores.argsort(descending=True)[:max_cands]
    b, s, l = boxes[order], scores[order], labels[order]
    iou = box_iou(b, b).triu(1) * (l[:, None] == l[None, :])
    comp = iou.max(0).values                                  # max IoU of each box with higher-scored ones
    decay = torch.exp(-sigma * (iou ** 2 - comp[:, None] ** 2)).min(0).values
    return order, s * decay


def merge(views: Dict, view_keys: Sequence, setting: dict, fields: Sequence[str]) -> Dict[str, np.ndarray]:
    parts = [views[k] for k in view_keys]
    det = {f: np.concatenate([p[f] for p in parts]) for f in fields}
    keep = det["scores"] >= setting["score_thresh"]
    det = {f: v[keep] for f, v in det.items()}
    if len(det["scores"]) == 0:
        return det
    boxes, scores = torch.from_numpy(det["boxes"]), torch.from_numpy(det["scores"])
    labels = torch.from_numpy(det["labels"])
    if setting["merge"] == "nms":
        idx = batched_nms(boxes, scores, labels, setting["nms_iou"])[: setting["max_dets"]].numpy()
        return {f: v[idx] for f, v in det.items()}
    order, new_scores = matrix_nms(boxes, scores, labels, setting["sigma"])
    ok = new_scores >= setting["score_thresh"]
    order, new_scores = order[ok], new_scores[ok]
    top = new_scores.argsort(descending=True)[: setting["max_dets"]]
    idx = order[top].numpy()
    out = {f: v[idx] for f, v in det.items()}
    out["scores"] = new_scores[top].numpy()
    return out


def view_keys_for(setting: dict) -> List[Tuple[float, bool]]:
    return [(s, f) for s in setting["scales"] for f in ((False, True) if setting["flip"] else (False,))]


def describe(setting: dict) -> str:
    views = "+".join(f"{s:g}x" for s in setting["scales"]) + (" +flip" if setting["flip"] else "")
    m = f"NMS {setting['nms_iou']:.2f}" if setting["merge"] == "nms" else f"Matrix-NMS s={setting['sigma']:g}"
    return f"{views:<20} {m:<20} thr {setting['score_thresh']:.2f}"


DEFAULT_SETTING = {"scales": [1.0], "flip": False, "merge": "nms", "nms_iou": DEFAULT_INFERENCE["nms_iou"],
                   "sigma": None, "score_thresh": DEFAULT_INFERENCE["score_thresh"],
                   "max_dets": DEFAULT_INFERENCE["max_dets"]}


# --------------------------------------------------------------------------- phases

def collect(model, loader, indices, scales, flip, device, seed, fields, desc):
    """Runs every view on every image; yields (position, views, gt)."""
    locations = LocationCache(model.det_head)
    t0 = time.time()
    for i, batch in enumerate(loader):
        gt = {"image_id": batch["image_ids"][0], "boxes": batch["boxes"][0].numpy(),
              "labels": batch["labels"][0].numpy()}
        if "attributes" in batch:
            gt["attributes"] = batch["attributes"][0]
        views = run_views(model, locations, batch["images"][0], scales, flip, device, seed, indices[i],
                          batch["ignore_masks"][0].numpy(), fields)
        yield i, views, gt
        if (i + 1) % 50 == 0 or i + 1 == len(loader):
            el = time.time() - t0
            eta = el / (i + 1) * (len(loader) - i - 1)
            print(f"[{desc}] {i + 1}/{len(loader)}  {el / 60:.1f} min elapsed, ~{eta / 60:.1f} min left", flush=True)


def phase_tune(args, model, meta, exp_cfg, device, img_size):
    stride = args.stride or 30
    ds, indices, folds, num_classes, _ = build_split(exp_cfg, img_size, stride, args.max_images)
    img_fold = np.array([folds[ds.frames[i][0]] for i in indices])
    print(f"[TUNE] {len(indices)} test frames (stride {stride}); fold sizes "
          f"{int((img_fold == 0).sum())} / {int((img_fold == 1).sum())} images, "
          f"{sum(v == 0 for v in folds.values())} / {sum(v == 1 for v in folds.values())} videos")
    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, f"{args.name}_tune_cache_s{stride}.pkl")
    scales = sorted(set([1.0] + args.scales))

    cache = None
    if os.path.exists(cache_path) and not args.recompute:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("weights") != meta["weights"] or cache.get("indices") != indices or cache.get("scales") != scales:
            print("[TUNE] cache is for different weights/images/scales; recomputing")
            cache = None
        elif len(cache["views"]) == len(indices):
            print(f"[TUNE] reusing cached forward passes: {cache_path}")
        else:
            print(f"[TUNE] resuming forward passes after {len(cache['views'])} images")
    if cache is None:
        cache = {"weights": meta["weights"], "indices": indices, "scales": scales, "views": [], "gts": []}

    def dump():
        with open(cache_path + ".tmp", "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(cache_path + ".tmp", cache_path)

    todo = indices[len(cache["views"]):]
    if todo:
        loader = DataLoader(Subset(ds, todo), batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)
        try:
            for _, views, gt in collect(model, loader, todo, scales, True, device, args.seed, FIELDS_TUNE, "TUNE"):
                cache["views"].append(views)
                cache["gts"].append(gt)
                if len(cache["views"]) % 50 == 0:
                    dump()
        finally:
            dump()   # keep finished images even after a crash or Ctrl+C; rerun the same command to resume

    all_views, gts = cache["views"], cache["gts"]
    extra = [s for s in scales if s != 1.0]
    scale_sets = [[1.0]] + [[1.0] + list(c) for r in range(1, len(extra) + 1) for c in itertools.combinations(extra, r)]
    grid = []
    for sc, fl, thr in itertools.product(scale_sets, (False, True), (0.02, 0.05)):
        for iou in (0.5, 0.6, 0.7):
            grid.append(dict(DEFAULT_SETTING, scales=sc, flip=fl, score_thresh=thr, nms_iou=iou))
        grid.append(dict(DEFAULT_SETTING, scales=sc, flip=fl, score_thresh=thr, merge="matrix", sigma=2.0, nms_iou=None))

    fold_idx = [np.where(img_fold == k)[0] for k in (0, 1)]
    rows, preds_by_setting = [], []
    print(f"[TUNE] scoring {len(grid)} settings (mAP on fold A | fold B | all)")
    for n, setting in enumerate(grid, 1):
        keys = view_keys_for(setting)
        preds = []
        for v in all_views:
            p = merge(v, keys, setting, FIELDS_TUNE)
            p["center_ignored"] = p.pop("ignored")
            preds.append(p)
        fold_map = [compute_map([preds[i] for i in idx], [gts[i] for i in idx], num_classes)["mAP"] for idx in fold_idx]
        m = compute_map(preds, gts, num_classes)
        cost = sum(s * s for s in setting["scales"]) * (2 if setting["flip"] else 1)
        rows.append({"setting": setting, "mAP": m["mAP"], "AP50": m["AP50"], "AP75": m["AP75"],
                     "mAP_fold": fold_map, "cost": cost})
        preds_by_setting.append(preds)
        print(f"  [{n:3d}/{len(grid)}] {describe(setting)}  {fold_map[0]:.4f} | {fold_map[1]:.4f} | "
              f"{m['mAP']:.4f}  ({cost:.2f}x)", flush=True)

    def choose(k):
        """Cheapest setting within `margin` mAP of the best on fold k (differences below that are noise)."""
        best = max(r["mAP_fold"][k] for r in rows)
        ok = [j for j, r in enumerate(rows) if r["mAP_fold"][k] >= best - args.margin]
        return min(ok, key=lambda j: (rows[j]["cost"], -rows[j]["mAP_fold"][k]))

    base = rows.index(next(r for r in rows if r["setting"] == DEFAULT_SETTING))
    if args.rule == "cv":
        pick = {0: choose(1), 1: choose(0)}  # images of fold k use the setting chosen on the other fold
    else:
        # robust: the one setting whose smaller per-fold gain over the default is largest, used for every
        # video. It looks at both folds, so the "cross-validated" line below is then not held out.
        gains = [min(r["mAP_fold"][k] - rows[base]["mAP_fold"][k] for k in (0, 1)) for r in rows]
        j = max(range(len(rows)), key=lambda j: (round(gains[j], 4), -rows[j]["cost"]))
        pick = {0: j, 1: j}
        print(f"\n[TUNE] robust rule: {describe(rows[j]['setting'])} improves both folds by >= {gains[j]:+.4f} mAP")
    cross = [preds_by_setting[pick[int(img_fold[i])]][i] for i in range(len(gts))]
    cv = compute_map(cross, gts, num_classes)

    print("\n[TUNE] top 10 settings on all tuning frames (for information only; not used for the choice)")
    for r in sorted(rows, key=lambda r: -r["mAP"])[:10]:
        print(f"  {describe(r['setting'])}  mAP {r['mAP']:.4f} ({r['mAP'] - rows[base]['mAP']:+.4f})  "
              f"AP50 {r['AP50']:.4f}  cost {r['cost']:.2f}x")
    print(f"\n[TUNE] chosen on fold B, used for fold A videos: {describe(rows[pick[0]]['setting'])}")
    print(f"[TUNE] chosen on fold A, used for fold B videos: {describe(rows[pick[1]]['setting'])}")
    print(f"[TUNE] default                 mAP {rows[base]['mAP']:.4f}  AP50 {rows[base]['AP50']:.4f}")
    print(f"[TUNE] cross-validated tuned   mAP {cv['mAP']:.4f}  AP50 {cv['AP50']:.4f}  "
          f"({cv['mAP'] - rows[base]['mAP']:+.4f} mAP)" + ("  <- held-out estimate" if args.rule == "cv" else "  (not held out)"))

    choice = {"folds": folds, "by_fold": {"0": rows[pick[0]]["setting"], "1": rows[pick[1]]["setting"]},
              "tuned_on": {"stride": stride, "num_images": len(indices), "margin": args.margin, "rule": args.rule}}
    result = {"meta": {**meta, "phase": "tune", "stride": stride, "num_images": len(indices), "git": git_revision(),
                       "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
              "default": rows[base], "cross_validated": {"mAP": cv["mAP"], "AP50": cv["AP50"], "AP75": cv["AP75"]},
              "choice": choice, "rows": rows}
    out = os.path.join(args.out_dir, f"{args.name}_tune.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    path = os.path.join(args.out_dir, f"{args.name}_choice.json")
    with open(path, "w") as f:
        json.dump(choice, f, indent=2)
    print(f"\n[TUNE] wrote {out}\n[TUNE] wrote {path}")


def phase_test(args, model, meta, exp_cfg, device, img_size):
    with open(args.choice) as f:
        choice = json.load(f)
    stride = args.stride or 1
    ds, indices, folds, num_classes, names = build_split(exp_cfg, img_size, stride, args.max_images, args.offset)
    tuned_frames = set(range(0, len(ds), choice["tuned_on"]["stride"]))
    overlap = sum(i in tuned_frames for i in indices)
    if overlap:
        print(f"[TEST] warning: {overlap} of {len(indices)} frames were used for tuning (use --offset to avoid them)")
    if folds != choice["folds"]:
        raise SystemExit("fold assignment differs from the one used for tuning")
    by_fold = {int(k): v for k, v in choice["by_fold"].items()}
    scales = sorted(set([1.0] + by_fold[0]["scales"] + by_fold[1]["scales"]))
    flip = by_fold[0]["flip"] or by_fold[1]["flip"]
    print(f"[TEST] {len(indices)} test frames (stride {stride}), views {scales}{' +flip' if flip else ''}")
    for k in (0, 1):
        print(f"[TEST] fold {'AB'[k]} videos use: {describe(by_fold[k])}")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.name}_test_s{stride}" + (f"o{args.offset}" if args.offset else "")
    partial = os.path.join(args.out_dir, f"{tag}_partial.pkl")

    state = {"done": 0, "default": [], "tuned": [], "gts": []}
    if os.path.exists(partial):
        with open(partial, "rb") as f:
            saved = pickle.load(f)
        if saved.get("choice") == choice and saved.get("weights") == meta["weights"] and saved.get("indices") == indices:
            state = saved
            print(f"[TEST] resuming after {state['done']} images")
    todo = indices[state["done"]:]
    loader = DataLoader(Subset(ds, todo), batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

    def dump():
        state.update(choice=choice, weights=meta["weights"], indices=indices)
        with open(partial + ".tmp", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(partial + ".tmp", partial)

    try:
        for i, views, gt in collect(model, loader, todo, scales, flip, device, args.seed, FIELDS_FULL, "TEST"):
            tuned = by_fold[folds[ds.frames[todo[i]][0]]]
            for key, setting in (("default", DEFAULT_SETTING), ("tuned", tuned)):
                p = merge(views, view_keys_for(setting), setting, FIELDS_FULL)
                p["center_ignored"] = p.pop("ignored")
                p["image_id"] = gt["image_id"]
                state[key].append(p)
            state["gts"].append(gt)
            state["done"] += 1
            if state["done"] % 1000 == 0:
                dump()
    except KeyboardInterrupt:
        dump()
        raise SystemExit("\n[TEST] interrupted; progress saved, rerun the same command to resume")
    except Exception:
        dump()
        print("[TEST] crashed; progress saved, rerun the same command to resume", flush=True)
        raise

    results = {key: summarize(state[key], state["gts"], num_classes) for key in ("default", "tuned")}
    d, t = results["default"], results["tuned"]
    print("\n[TEST] metric      default   tuned     change")
    for k in ("mAP", "AP50", "AP75", "APs", "APm", "APl", "D-ECE"):
        if isinstance(d.get(k), (float, np.floating)) and isinstance(t.get(k), (float, np.floating)):
            print(f"       {k:<10} {d[k]:.4f}    {t[k]:.4f}    {t[k] - d[k]:+.4f}")
    out = os.path.join(args.out_dir, f"{tag}.json")
    with open(out, "w") as f:
        json.dump({"meta": {**meta, "phase": "test", "stride": stride, "offset": args.offset, "tuning_frames_included": overlap, "num_images": len(indices),
                            "dataset_size": len(ds), "class_names": names, "choice": choice,
                            "default": DEFAULT_SETTING, "git": git_revision(), "device": str(device),
                            "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
                   "metrics": results}, f, indent=2, default=float)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"[TEST] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["tune", "test"])
    ap.add_argument("--weights", nargs="+", required=True, help="one checkpoint, or several to average")
    ap.add_argument("--exp", default="configs/experiments/uavdt_stage4_residual.yaml")
    ap.add_argument("--choice", help="test phase: JSON setting written by the tune phase")
    ap.add_argument("--scales", nargs="+", type=float, default=[1.25, 1.5], help="tune: extra scales besides 1.0")
    ap.add_argument("--stride", type=int, default=0,
                    help="every Nth test frame (default: 30 for tune, 1 = full test set for test)")
    ap.add_argument("--rule", choices=["cv", "robust"], default="cv",
                    help="tune: cv = 2-fold cross-tuning; robust = one setting that helps both folds most")
    ap.add_argument("--margin", type=float, default=0.002,
                    help="tune: prefer the cheapest setting within this mAP of the best (noise level)")
    ap.add_argument("--offset", type=int, default=0, help="test: first frame index (stride 5 offset 2 skips all tuning frames)")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="output file prefix (default: checkpoint name)")
    ap.add_argument("--out-dir", default="results/tune")
    ap.add_argument("--recompute", action="store_true", help="tune: ignore the cached forward passes")
    args = ap.parse_args()
    if args.phase == "test" and not args.choice:
        ap.error("test phase needs --choice")
    if args.name is None:
        stem = os.path.splitext(os.path.basename(args.weights[0]))[0]
        args.name = stem + (f"_avg{len(args.weights)}" if len(args.weights) > 1 else "")

    device = pick_device(args.device)
    exp_cfg = load_yaml(args.exp)
    img_size = tuple(exp_cfg["train"].get("img_size") or load_yaml(exp_cfg["dataset_config"])["image_size"])
    model, meta = load_model(args.weights, device)
    print(f"[MODEL] {meta['weights']}  epoch {meta['epoch']} stage {meta['stage']}  device {device}  input {img_size}")
    (phase_tune if args.phase == "tune" else phase_test)(args, model, meta, exp_cfg, device, img_size)


if __name__ == "__main__":
    main()
