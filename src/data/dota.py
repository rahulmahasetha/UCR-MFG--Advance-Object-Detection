import os
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from PIL import Image

from src.data.dataset import BaseDataset
from src.data.transforms import resize_and_pad, transform_boxes, transform_ignore_mask


class DOTADataset(BaseDataset):
    """DOTA-v1.5 aerial dataset loader with documented OBB-to-HBB envelope conversion.
    
    Converts 8-coordinate oriented bounding boxes into horizontal bounding boxes:
        [x_min, y_min, x_max, y_max]
    Note: Converted HBB evaluations are internal diagnostics and must not be compared
    directly to official oriented leaderboard benchmarks.
    """

    CLASSES = [
        "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
        "basketball-court", "ground-track-field", "harbor", "bridge",
        "large-vehicle", "small-vehicle", "helicopter", "roundabout",
        "soccer-ball-field", "swimming-pool", "container-crane"
    ]

    def __init__(
        self,
        root_dir: str = "dataset/DOTA-v1.5",
        split: str = "validation",
        img_size: Tuple[int, int] = (1024, 1024),
        is_train: bool = False
    ):
        super().__init__(root_dir, split, img_size, is_train)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.CLASSES)}
        self.img_dir = os.path.join(root_dir, split, "images")
        self.anno_dir = os.path.join(root_dir, split, "labelTxt")

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"DOTA images directory not found at: {self.img_dir}")

        image_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.samples: List[Tuple[str, str]] = []
        for img_name in image_files:
            stem = os.path.splitext(img_name)[0]
            anno_path = os.path.join(self.anno_dir, f"{stem}.txt")
            self.samples.append((os.path.join(self.img_dir, img_name), anno_path))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def obb_to_hbb(coords: List[float]) -> Tuple[float, float, float, float]:
        """Projects 8-point OBB vertices to horizontal bounding envelope [xmin, ymin, xmax, ymax]."""
        xs = coords[0::2]
        ys = coords[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, anno_path = self.samples[idx]
        image_id = os.path.splitext(os.path.basename(img_path))[0]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            image_tensor, scale, padding = resize_and_pad(img, self.img_size)

        boxes_list = []
        labels_list = []
        ignore_mask_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if os.path.exists(anno_path):
            with open(anno_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    try:
                        coords = [float(p) for p in parts[:8]]
                        cat_name = parts[8].strip()
                        difficult = int(parts[9]) if len(parts) > 9 else 0
                    except (ValueError, IndexError):
                        continue

                    xmin, ymin, xmax, ymax = self.obb_to_hbb(coords)
                    x1 = max(0.0, xmin)
                    y1 = max(0.0, ymin)
                    x2 = min(float(orig_w), xmax)
                    y2 = min(float(orig_h), ymax)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    if difficult == 1 or cat_name not in self.class_to_idx:
                        ignore_mask_orig[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))] = 1
                    else:
                        boxes_list.append([x1, y1, x2, y2])
                        labels_list.append(self.class_to_idx[cat_name])

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
            "ignore_mask": ignore_mask,
            "orig_size": (orig_h, orig_w),
            "scale_factor": (scale, scale),
        }

