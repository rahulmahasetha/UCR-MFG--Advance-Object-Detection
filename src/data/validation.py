import os
from typing import Dict, Any, List, Set, Tuple
import torch
from src.data.dataset import BaseDataset


class DataValidator:
    """10-Point Data Integrity and Validation Engine."""

    def __init__(self, num_classes: int = 10, img_size: Tuple[int, int] = (1024, 1024)):
        self.num_classes = num_classes
        self.img_size = img_size

    def validate_dataset(self, dataset: BaseDataset, max_samples: int = 100) -> Dict[str, Any]:
        """Runs the complete 10-point audit over dataset samples."""
        results = {
            "dataset_name": dataset.__class__.__name__,
            "total_samples": len(dataset),
            "checked_samples": min(len(dataset), max_samples),
            "errors": [],
            "warnings": [],
            "degenerate_boxes": 0,
            "out_of_bounds_boxes": 0,
            "duplicate_boxes": 0,
            "unknown_classes": 0,
            "empty_annotation_images": 0,
            "valid": True,
        }

        checked_count = min(len(dataset), max_samples)
        seen_image_ids: Set[str] = set()

        for idx in range(checked_count):
            try:
                sample = dataset[idx]
            except Exception as e:
                results["errors"].append(f"Sample {idx} failed to load: {e}")
                results["valid"] = False
                continue

            image_id = sample["image_id"]
            if image_id in seen_image_ids:
                results["warnings"].append(f"Duplicate image_id detected: {image_id}")
            seen_image_ids.add(image_id)

            image = sample["image"]
            boxes = sample["boxes"]
            labels = sample["labels"]
            ignore_mask = sample["ignore_mask"]

            # 1. Image tensor sanity
            if image.shape != (3, self.img_size[0], self.img_size[1]):
                results["errors"].append(f"Sample {image_id}: Invalid image shape {image.shape}")
                results["valid"] = False

            # 2. Ignore mask sanity
            if ignore_mask.shape != (self.img_size[0], self.img_size[1]):
                results["errors"].append(f"Sample {image_id}: Invalid ignore mask shape {ignore_mask.shape}")
                results["valid"] = False

            # 3. Box count matches label count
            if len(boxes) != len(labels):
                results["errors"].append(f"Sample {image_id}: Box count {len(boxes)} != Label count {len(labels)}")
                results["valid"] = False

            if len(boxes) == 0:
                results["empty_annotation_images"] += 1
                continue

            # 4. Box geometry & boundaries
            target_h, target_w = self.img_size
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

            degenerate = (x2 <= x1) | (y2 <= y1)
            if degenerate.any():
                count = degenerate.sum().item()
                results["degenerate_boxes"] += count
                results["errors"].append(f"Sample {image_id}: {count} degenerate boxes detected.")
                results["valid"] = False

            oob = (x1 < 0) | (y1 < 0) | (x2 > target_w) | (y2 > target_h)
            if oob.any():
                count = oob.sum().item()
                results["out_of_bounds_boxes"] += count
                results["warnings"].append(f"Sample {image_id}: {count} out-of-bounds boxes detected.")

            # 5. Class ID bounds
            invalid_classes = (labels < 0) | (labels >= self.num_classes)
            if invalid_classes.any():
                count = invalid_classes.sum().item()
                results["unknown_classes"] += count
                results["errors"].append(f"Sample {image_id}: {count} labels outside [0, {self.num_classes - 1}].")
                results["valid"] = False

            # 6. Duplicate annotations
            if len(boxes) > 1:
                stacked = torch.cat([boxes, labels.unsqueeze(1).float()], dim=1)
                unique_len = len(torch.unique(stacked, dim=0))
                diff = len(boxes) - unique_len
                if diff > 0:
                    results["duplicate_boxes"] += diff
                    results["warnings"].append(f"Sample {image_id}: {diff} duplicate annotations found.")

        return results

    @staticmethod
    def verify_split_isolation(train_ids: Set[str], val_ids: Set[str], test_ids: Set[str]) -> bool:
        """Asserts zero overlap between train, val, and test splits."""
        train_val = train_ids.intersection(val_ids)
        train_test = train_ids.intersection(test_ids)
        val_test = val_ids.intersection(test_ids)

        if train_val or train_test or val_test:
            raise ValueError(
                f"Data Leakage Violation! Overlaps: "
                f"train-val={len(train_val)}, train-test={len(train_test)}, val-test={len(val_test)}"
            )
        return True

