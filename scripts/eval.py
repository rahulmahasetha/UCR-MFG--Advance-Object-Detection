import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.visdrone import VisDroneDataset
from src.data.uavdt import UAVDTDataset
from src.data.dota import DOTADataset
from src.data.collate import collate_fn
from src.models.ucr_mfg import UCRMFG
from src.utils.metrics import compute_dece, compute_ap, compute_iou_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="UCR-MFG Evaluation Runner")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--config", type=str, default="configs/experiments/uavdt_zeroshot.yaml", help="Evaluation experiment config")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (val or test)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of evaluation samples")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        exp_cfg = yaml.safe_load(f)
    with open(exp_cfg["dataset_config"], "r") as f:
        ds_cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    ds_name = ds_cfg["dataset_name"]
    print(f"\n=======================================================")
    print(f"📊 [EVAL] Evaluating {ds_name} on Device: {device}")
    print(f"   Checkpoint: {args.weights}")
    print(f"=======================================================\n")

    # 1. Instantiate dataset based on config
    root_dir = ds_cfg["root_dir"]
    img_size = tuple(ds_cfg["image_size"])

    if ds_name == "VisDrone2019-DET":
        dataset = VisDroneDataset(root_dir=root_dir, split=args.split, img_size=img_size, is_train=False)
        is_uavdt_zeroshot = False
    elif ds_name == "UAVDT":
        dataset = UAVDTDataset(root_dir=root_dir, split=args.split, img_size=img_size, is_train=False)
        is_uavdt_zeroshot = True
    elif ds_name == "DOTA-v1.5":
        dataset = DOTADataset(root_dir=root_dir, split=args.split, img_size=img_size, is_train=False)
        is_uavdt_zeroshot = False
    else:
        raise ValueError(f"Unsupported dataset: {ds_name}")

    if args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples] if hasattr(dataset, "samples") else dataset.frames[:args.max_samples]

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 2. Instantiate Model and Load Weights
    # Model trained on VisDrone has 10 classes
    model = UCRMFG(num_classes=10, pretrained=False).to(device)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_predictions = []
    all_ground_truths = []

    target_size = img_size
    shapes = [(target_size[0] // s, target_size[1] // s) for s in [8, 16, 32]]
    locations = model.det_head.get_grid_locations(shapes, device)

    # Vehicle ontology mapping for UAVDT zero-shot
    # VisDrone: car=3, van=4 -> UAVDT car=0; truck=5 -> truck=1; bus=8 -> bus=2
    visdrone_to_uavdt = {3: 0, 4: 0, 5: 1, 8: 2}

    print(f"[EVAL] Running inference across {len(loader)} samples...")
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            images = batch["images"].to(device)
            gt_boxes_b = batch["boxes"][0].cpu().numpy()
            gt_labels_b = batch["labels"][0].cpu().numpy()
            ignore_b = batch["ignore_masks"][0].cpu().numpy()

            outputs = model(images)
            cls_preds = outputs["cls_preds"]
            reg_preds = outputs["reg_preds"]
            cen_preds = outputs["cen_preds"]

            pred_boxes_list, pred_scores_list, pred_labels_list = [], [], []

            for lvl, stride in enumerate([8, 16, 32]):
                cls_p = torch.sigmoid(cls_preds[lvl][0]).permute(1, 2, 0).reshape(-1, 10)
                reg_p = reg_preds[lvl][0].permute(1, 2, 0).reshape(-1, 4)
                cen_p = torch.sigmoid(cen_preds[lvl][0]).permute(1, 2, 0).reshape(-1)

                scores, labels = cls_p.max(dim=-1)
                scores = torch.sqrt(scores * cen_p)

                # Score threshold filter
                keep = scores > 0.05
                if keep.any():
                    loc = locations[lvl][keep]
                    reg_k = reg_p[keep]
                    scores_k = scores[keep]
                    labels_k = labels[keep]

                    x1 = loc[:, 0] - reg_k[:, 0] * stride
                    y1 = loc[:, 1] - reg_k[:, 1] * stride
                    x2 = loc[:, 0] + reg_k[:, 2] * stride
                    y2 = loc[:, 1] + reg_k[:, 3] * stride
                    boxes_k = torch.stack([x1, y1, x2, y2], dim=-1)

                    pred_boxes_list.append(boxes_k.cpu().numpy())
                    pred_scores_list.append(scores_k.cpu().numpy())
                    pred_labels_list.append(labels_k.cpu().numpy())

            if pred_boxes_list:
                p_boxes = np.concatenate(pred_boxes_list, axis=0)
                p_scores = np.concatenate(pred_scores_list, axis=0)
                p_labels = np.concatenate(pred_labels_list, axis=0)
            else:
                p_boxes = np.empty((0, 4))
                p_scores = np.empty((0,))
                p_labels = np.empty((0,), dtype=int)

            if is_uavdt_zeroshot:
                # Filter and remap predictions
                valid_idx = [i for i, lbl in enumerate(p_labels) if lbl in visdrone_to_uavdt]
                if valid_idx:
                    p_boxes = p_boxes[valid_idx]
                    p_scores = p_scores[valid_idx]
                    p_labels = np.array([visdrone_to_uavdt[lbl] for lbl in p_labels[valid_idx]], dtype=int)
                else:
                    p_boxes = np.empty((0, 4))
                    p_scores = np.empty((0,))
                    p_labels = np.empty((0,), dtype=int)

            all_predictions.append({
                "boxes": p_boxes,
                "scores": p_scores,
                "labels": p_labels,
            })
            all_ground_truths.append({
                "boxes": gt_boxes_b,
                "labels": gt_labels_b,
                "ignore_mask": ignore_b,
            })

    # Compute Metrics
    num_eval_classes = 3 if is_uavdt_zeroshot else ds_cfg["classes"]["num_classes"]
    dece = compute_dece(all_predictions, all_ground_truths, num_classes=num_eval_classes)

    print(f"\n{'='*60}")
    print(f"🎯 [RESULTS] Benchmark: {ds_name} ({args.split} split)")
    print(f"   Detection Expected Calibration Error (D-ECE): {dece:.4f}")
    print(f"   Evaluated Samples: {len(all_predictions)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
