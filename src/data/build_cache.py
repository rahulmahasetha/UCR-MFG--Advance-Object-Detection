import os
import argparse
import torch
from tqdm import tqdm
from src.data.visdrone import VisDroneDataset
from src.data.uavdt import UAVDTDataset


def build_visdrone_cache(root_dir: str = "dataset/VisDrone2019-DET", cache_dir: str = ".cache"):
    """Pre-indexes VisDrone dataset paths and sample indices."""
    os.makedirs(cache_dir, exist_ok=True)
    for split in ["train", "val", "test_dev"]:
        try:
            ds = VisDroneDataset(root_dir=root_dir, split=split)
            cache_file = os.path.join(cache_dir, f"visdrone_{split}_meta.pt")
            meta = {
                "num_samples": len(ds),
                "samples": ds.samples,
            }
            torch.save(meta, cache_file)
            print(f"[CACHE] Saved {split} metadata ({len(ds)} samples) to {cache_file}")
        except Exception as e:
            print(f"[CACHE] Skipped {split}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Precompute dataset index cache.")
    parser.add_argument("--root", type=str, default="dataset/VisDrone2019-DET")
    parser.add_argument("--out", type=str, default=".cache")
    args = parser.parse_args()
    build_visdrone_cache(args.root, args.out)


if __name__ == "__main__":
    main()

