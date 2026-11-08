import os
import torch
import numpy as np
import pytest
from PIL import Image

from src.data.dataset import BaseDataset
from src.data.transforms import resize_and_pad, transform_boxes, transform_ignore_mask
from src.data.augmentations import TrainAugmenter
from src.data.collate import collate_fn
from src.data.visdrone import VisDroneDataset
from src.data.uavdt import UAVDTDataset
from src.data.dota import DOTADataset


def test_transforms_resize_and_pad():
    img = Image.new("RGB", (800, 600), color=(255, 0, 0))
    tensor, scale, padding = resize_and_pad(img, target_size=(1024, 1024))
    assert tensor.shape == (3, 1024, 1024)
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    assert scale == 1024 / 800  # 1.28
    pad_w, pad_h = padding
    assert pad_w == 0
    assert pad_h == (1024 - int(round(600 * 1.28))) // 2


def test_transforms_box_and_mask():
    target_size = (1024, 1024)
    boxes = torch.tensor([[100.0, 100.0, 200.0, 300.0]])
    scale = 1.0
    padding = (10, 20)
    transformed = transform_boxes(boxes, scale, padding, target_size)
    assert transformed[0, 0] == 110.0
    assert transformed[0, 1] == 120.0
    assert transformed[0, 2] == 210.0
    assert transformed[0, 3] == 320.0

    mask_orig = np.zeros((600, 800), dtype=np.uint8)
    mask_orig[100:200, 100:200] = 1
    t_mask = transform_ignore_mask(mask_orig, scale, padding, target_size)
    assert t_mask.shape == target_size
    assert t_mask.dtype == torch.uint8


def test_augmentations_flip():
    img = torch.rand(3, 128, 128)
    boxes = torch.tensor([[10.0, 20.0, 40.0, 60.0]])
    mask = torch.zeros(128, 128, dtype=torch.uint8)

    augmenter = TrainAugmenter(flip_p=1.0, jitter_p=0.0)
    aug_img, aug_boxes, aug_mask = augmenter(img, boxes, mask)

    # Box x-coordinates should be flipped: [128 - 40, 20, 128 - 10, 60] = [88, 20, 118, 60]
    assert torch.isclose(aug_boxes[0, 0], torch.tensor(88.0))
    assert torch.isclose(aug_boxes[0, 2], torch.tensor(118.0))


def test_collate_fn():
    batch = [
        {
            "image_id": "test_1",
            "image": torch.rand(3, 256, 256),
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
            "ignore_mask": torch.zeros(256, 256, dtype=torch.uint8),
            "orig_size": (400, 400),
            "scale_factor": (0.64, 0.64),
        },
        {
            "image_id": "test_2",
            "image": torch.rand(3, 256, 256),
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.int64),
            "ignore_mask": torch.zeros(256, 256, dtype=torch.uint8),
            "orig_size": (500, 500),
            "scale_factor": (0.512, 0.512),
        }
    ]
    collated = collate_fn(batch)
    assert collated["images"].shape == (2, 3, 256, 256)
    assert collated["ignore_masks"].shape == (2, 256, 256)
    assert len(collated["boxes"]) == 2
    assert len(collated["labels"]) == 2
    assert collated["image_ids"] == ["test_1", "test_2"]


def test_dota_obb_to_hbb():
    # 4 corners of an oriented rectangle
    coords = [10.0, 20.0, 50.0, 10.0, 60.0, 40.0, 20.0, 50.0]
    xmin, ymin, xmax, ymax = DOTADataset.obb_to_hbb(coords)
    assert xmin == 10.0
    assert ymin == 10.0
    assert xmax == 60.0
    assert ymax == 50.0

