import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
from src.data.visdrone import VisDroneDataset
from src.data.uavdt import UAVDTDataset
from src.data.dota import DOTADataset
from src.data.validation import DataValidator


def main():
    parser = argparse.ArgumentParser(description="UCR-MFG 10-Point Data Validation Runner")
    parser.add_argument("--config", type=str, default="configs/datasets/visdrone.yaml", help="Path to dataset YAML config")
    parser.add_argument("--split", type=str, default="train", help="Split to audit (train, val, test)")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to audit")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ds_name = cfg["dataset_name"]
    root_dir = cfg["root_dir"]
    img_size = tuple(cfg["image_size"])
    num_classes = cfg["classes"]["num_classes"]

    print(f"\n{'='*70}")
    print(f"🔍 [AUDIT] Running 10-Point Data Validation on {ds_name} ({args.split} split)")
    print(f"{'='*70}")

    if ds_name == "VisDrone2019-DET":
        ds = VisDroneDataset(root_dir=root_dir, split=args.split, img_size=img_size, is_train=(args.split == "train"))
    elif ds_name == "UAVDT":
        ds = UAVDTDataset(root_dir=root_dir, split=args.split, img_size=img_size, is_train=(args.split == "train"))
    elif ds_name == "DOTA-v1.5":
        ds = DOTADataset(root_dir=root_dir, split=args.split, img_size=img_size, is_train=(args.split == "train"))
    else:
        raise ValueError(f"Unknown dataset {ds_name}")

    validator = DataValidator(num_classes=num_classes, img_size=img_size)
    results = validator.validate_dataset(ds, max_samples=args.samples)

    print(f"Total dataset size:        {results['total_samples']}")
    print(f"Samples audited:           {results['checked_samples']}")
    print(f"Degenerate boxes found:    {results['degenerate_boxes']}")
    print(f"Out-of-bounds boxes found: {results['out_of_bounds_boxes']}")
    print(f"Duplicate boxes found:     {results['duplicate_boxes']}")
    print(f"Unknown classes:           {results['unknown_classes']}")
    print(f"Empty annotation images:   {results['empty_annotation_images']}")
    print(f"Validation Status:         {'✅ PASSED (Clean)' if results['valid'] else '❌ FAILED'}")

    if results["errors"]:
        print(f"\nErrors (first 5):")
        for err in results["errors"][:5]:
            print(f"  ❌ {err}")

    if results["warnings"]:
        print(f"\nWarnings (first 5):")
        for warn in results["warnings"][:5]:
            print(f"  ⚠️ {warn}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
