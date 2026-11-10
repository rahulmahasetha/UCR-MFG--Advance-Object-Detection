import os
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from PIL import Image

from src.data.dataset import BaseDataset
from src.data.transforms import resize_and_pad, transform_boxes, transform_ignore_mask
from src.data.augmentations import TrainAugmenter


class VisDroneDataset(BaseDataset):
    """VisDrone2019-DET primary benchmark dataset loader.
    
    Parses official annotations:
        bbox_left, bbox_top, width, height, score, category, truncation, occlusion
    Extracts category == 0 and score == 0 into a spatial ignore mask.
    """

    SPLIT_MAP = {
        "train": "VisDrone2019-DET-train",
        "val": "VisDrone2019-DET-val",
        "test_dev": "VisDrone2019-DET-test-dev",
    }

    def __init__(
        self,
        root_dir: str = "dataset/VisDrone2019-DET",
        split: str = "train",
        img_size: Tuple[int, int] = (1024, 1024),
        is_train: bool = False
    ):
        super().__init__(root_dir, split, img_size, is_train)
        split_dir = self.SPLIT_MAP.get(split, split)
        self.img_dir = os.path.join(root_dir, split_dir, "images")
        self.anno_dir = os.path.join(root_dir, split_dir, "annotations")
        self.augmenter = TrainAugmenter() if is_train else None

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"VisDrone images directory not found at: {self.img_dir}")

        image_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.samples: List[Tuple[str, str]] = []
        for img_name in image_files:
            stem = os.path.splitext(img_name)[0]
            anno_name = f"{stem}.txt"
            anno_path = os.path.join(self.anno_dir, anno_name)
            self.samples.append((os.path.join(self.img_dir, img_name), anno_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, anno_path = self.samples[idx]
        image_id = os.path.splitext(os.path.basename(img_path))[0]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            image_tensor, scale, padding = resize_and_pad(img, self.img_size)

        boxes_list = []
        labels_list = []
        occlusion_list = []
        ignore_mask_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if os.path.exists(anno_path):
            with open(anno_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 8:
                        continue
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        w = float(parts[2])
                        h = float(parts[3])
                        score = int(parts[4])
                        category = int(parts[5])
                        occlusion = int(parts[7])
                    except ValueError:
                        continue

                    # Bounding coordinates
                    x1 = max(0.0, x)
                    y1 = max(0.0, y)
                    x2 = min(float(orig_w), x + w)
                    y2 = min(float(orig_h), y + h)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Ignore category 0 or score 0
                    if category == 0 or score == 0:
                        ix1, iy1 = int(round(x1)), int(round(y1))
                        ix2, iy2 = int(round(x2)), int(round(y2))
                        ignore_mask_orig[iy1:iy2, ix1:ix2] = 1
                    elif 1 <= category <= 10:
                        boxes_list.append([x1, y1, x2, y2])
                        labels_list.append(category - 1)  # 0-indexed: 0..9
                        occlusion_list.append(occlusion)

        if boxes_list:
            raw_boxes = torch.tensor(boxes_list, dtype=torch.float32)
            raw_labels = torch.tensor(labels_list, dtype=torch.int64)
            boxes = transform_boxes(raw_boxes, scale, padding, self.img_size)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            raw_labels = torch.empty((0,), dtype=torch.int64)

        ignore_mask = transform_ignore_mask(ignore_mask_orig, scale, padding, self.img_size)

        if self.augmenter is not None:
            image_tensor, boxes, ignore_mask = self.augmenter(image_tensor, boxes, ignore_mask)

        return {
            "image_id": image_id,
            "image": image_tensor,
            "boxes": boxes,
            "labels": raw_labels,
            "occlusion": torch.tensor(occlusion_list, dtype=torch.int64),
            "ignore_mask": ignore_mask,
            "orig_size": (orig_h, orig_w),
            "scale_factor": (scale, scale),
        }

