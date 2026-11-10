import os
import json
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from PIL import Image

from src.data.dataset import BaseDataset
from src.data.transforms import resize_and_pad, transform_boxes, transform_ignore_mask


class SODAADataset(BaseDataset):
    """SODA-A aerial small-object evaluation dataset loader.
    
    Loads images from train/test directories and parses COCO-style JSON annotations.
    Provides clear error diagnostics if annotations are pending extraction.
    """

    CLASSES = [
        "airplane", "helicopter", "small-vehicle", "large-vehicle",
        "ship", "harbor", "swimming-pool", "storage-tank", "container"
    ]

    def __init__(
        self,
        root_dir: str = "dataset/SODA-A",
        split: str = "train",
        img_size: Tuple[int, int] = (1024, 1024),
        is_train: bool = False
    ):
        super().__init__(root_dir, split, img_size, is_train)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.CLASSES)}
        self.img_dir = os.path.join(root_dir, split)
        self.anno_json_path = os.path.join(root_dir, f"{split}.json")

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"SODA-A images directory not found at: {self.img_dir}")

        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        # Attempt to load COCO-format JSON if present
        self.annotations: Dict[str, List[Dict[str, Any]]] = {}
        if os.path.exists(self.anno_json_path):
            with open(self.anno_json_path, "r") as f:
                data = json.load(f)
                img_id_to_file = {img["id"]: img["file_name"] for img in data.get("images", [])}
                for ann in data.get("annotations", []):
                    f_name = img_id_to_file.get(ann.get("image_id"))
                    if f_name:
                        self.annotations.setdefault(f_name, []).append(ann)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_name = self.image_files[idx]
        image_id = os.path.splitext(file_name)[0]
        img_path = os.path.join(self.img_dir, file_name)

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            image_tensor, scale, padding = resize_and_pad(img, self.img_size)

        boxes_list = []
        labels_list = []
        ignore_mask_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)

        anns = self.annotations.get(file_name, [])
        for ann in anns:
            bbox = ann.get("bbox", [])  # [x, y, w, h]
            if len(bbox) == 4:
                x, y, w, h = bbox
                x1, y1 = max(0.0, float(x)), max(0.0, float(y))
                x2, y2 = min(float(orig_w), x1 + float(w)), min(float(orig_h), y1 + float(h))
                cat_id = ann.get("category_id", 1) - 1  # 0-indexed
                if x2 > x1 and y2 > y1 and 0 <= cat_id < len(self.CLASSES):
                    boxes_list.append([x1, y1, x2, y2])
                    labels_list.append(cat_id)

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

