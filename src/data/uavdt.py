import os
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from PIL import Image

from src.data.dataset import BaseDataset
from src.data.transforms import resize_and_pad, transform_boxes, transform_ignore_mask


ATTRIBUTE_GROUPS = {
    "weather": ("daylight", "night", "fog"),
    "altitude": ("low", "medium", "high"),
    "view": ("front", "side", "bird"),
}


def parse_attributes(path: str) -> Dict[str, Any]:
    """UAVDT M_attr flags: daylight, night, fog, low/medium/high altitude, front/side/bird view, long-term."""
    with open(path) as f:
        flags = [int(v) for v in f.read().strip().split(",")]
    attributes: Dict[str, Any] = {"long_term": bool(flags[9]) if len(flags) > 9 else False}
    for g, (name, values) in enumerate(ATTRIBUTE_GROUPS.items()):
        attributes[name] = [v for v, flag in zip(values, flags[3 * g:3 * g + 3]) if flag]
    return attributes


class UAVDTDataset(BaseDataset):
    """UAVDT Benchmark cross-domain generalization evaluation dataset loader.
    
    Preserves official sequence-level splits and parses:
        <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<out-of-view>,<occlusion>,<object_category>
    Maps vehicle classes: car (1) -> 0, truck (2) -> 1, bus (3) -> 2.
    """

    def __init__(
        self,
        root_dir: str = "dataset/UAVDT-Benchmark-M",
        split: str = "test",
        img_size: Tuple[int, int] = (1024, 1024),
        is_train: bool = False
    ):
        super().__init__(root_dir, split, img_size, is_train)
        self.attr_dir = os.path.join(root_dir, "M_attr", split)
        self.images_root = os.path.join(root_dir, "UAV-benchmark-M")
        self.gt_root = os.path.join(root_dir, "UAV-benchmark-MOTD_v1.0", "GT")

        if not os.path.exists(self.attr_dir):
            raise FileNotFoundError(f"UAVDT split attribute directory not found at: {self.attr_dir}")

        seq_files = sorted([f for f in os.listdir(self.attr_dir) if f.endswith("_attr.txt")])
        self.sequence_names = [f.split("_attr.txt")[0].strip() for f in seq_files]
        self.sequence_attributes = {
            f.split("_attr.txt")[0].strip(): parse_attributes(os.path.join(self.attr_dir, f)) for f in seq_files
        }

        self.frames: List[Tuple[str, str, int]] = []  # (seq_name, img_path, frame_idx)
        self.annotations_cache: Dict[str, Dict[int, List[List[float]]]] = {}
        self.ignores_cache: Dict[str, Dict[int, List[List[float]]]] = {}

        for seq in self.sequence_names:
            seq_img_dir = os.path.join(self.images_root, seq)
            if not os.path.exists(seq_img_dir):
                continue

            # Load annotations
            gt_whole_path = os.path.join(self.gt_root, f"{seq}_gt_whole.txt")
            seq_annos: Dict[int, List[List[float]]] = {}
            if os.path.exists(gt_whole_path):
                with open(gt_whole_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) < 9:
                            continue
                        f_idx = int(parts[0])
                        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                        cat = int(parts[8])
                        if cat in (1, 2, 3):  # car, truck, bus
                            seq_annos.setdefault(f_idx, []).append([x, y, w, h, cat])
            self.annotations_cache[seq] = seq_annos

            # Load ignore regions
            gt_ignore_path = os.path.join(self.gt_root, f"{seq}_gt_ignore.txt")
            seq_ignores: Dict[int, List[List[float]]] = {}
            if os.path.exists(gt_ignore_path):
                with open(gt_ignore_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) < 6:
                            continue
                        f_idx = int(parts[0])
                        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                        seq_ignores.setdefault(f_idx, []).append([x, y, w, h])
            self.ignores_cache[seq] = seq_ignores

            # Index available frames
            frame_files = sorted([f for f in os.listdir(seq_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for ff in frame_files:
                # Frame index usually derived from filename e.g. img000001.jpg -> 1
                stem = os.path.splitext(ff)[0]
                try:
                    f_num = int("".join([c for c in stem if c.isdigit()]))
                except ValueError:
                    f_num = 0
                self.frames.append((seq, os.path.join(seq_img_dir, ff), f_num))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_name, img_path, frame_num = self.frames[idx]
        image_id = f"{seq_name}_{os.path.splitext(os.path.basename(img_path))[0]}"

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            image_tensor, scale, padding = resize_and_pad(img, self.img_size)

        boxes_list = []
        labels_list = []
        ignore_mask_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # Apply ignores
        for (ix, iy, iw, ih) in self.ignores_cache.get(seq_name, {}).get(frame_num, []):
            x1 = max(0.0, ix)
            y1 = max(0.0, iy)
            x2 = min(float(orig_w), ix + iw)
            y2 = min(float(orig_h), iy + ih)
            if x2 > x1 and y2 > y1:
                ignore_mask_orig[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))] = 1

        # Apply valid targets
        for (bx, by, bw, bh, cat) in self.annotations_cache.get(seq_name, {}).get(frame_num, []):
            x1 = max(0.0, bx)
            y1 = max(0.0, by)
            x2 = min(float(orig_w), bx + bw)
            y2 = min(float(orig_h), by + bh)
            if x2 > x1 and y2 > y1:
                boxes_list.append([x1, y1, x2, y2])
                labels_list.append(int(cat) - 1)  # 1->0 (car), 2->1 (truck), 3->2 (bus)

        if boxes_list:
            raw_boxes = torch.tensor(boxes_list, dtype=torch.float32)
            raw_labels = torch.tensor(labels_list, dtype=torch.int64)
            boxes = transform_boxes(raw_boxes, scale, padding, self.img_size)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            raw_labels = torch.empty((0,), dtype=torch.int64)

        ignore_mask = transform_ignore_mask(ignore_mask_orig, scale, padding, self.img_size)

        return {
            "image_id": image_id,
            "image": image_tensor,
            "boxes": boxes,
            "labels": raw_labels,
            "attributes": self.sequence_attributes.get(seq_name, {}),
            "ignore_mask": ignore_mask,
            "orig_size": (orig_h, orig_w),
            "scale_factor": (scale, scale),
        }

