import torch
import pytest
from src.data.validation import DataValidator


def test_split_isolation():
    train_ids = {"img1", "img2", "img3"}
    val_ids = {"img4", "img5"}
    test_ids = {"img6", "img7"}

    # Disjoint splits should pass
    assert DataValidator.verify_split_isolation(train_ids, val_ids, test_ids) is True

    # Leaking splits should raise ValueError
    leaking_val = {"img3", "img4"}
    with pytest.raises(ValueError):
        DataValidator.verify_split_isolation(train_ids, leaking_val, test_ids)


def test_validator_detects_degenerate_boxes():
    validator = DataValidator(num_classes=10, img_size=(256, 256))

    # Mock dataset object
    class DummyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "image_id": "dummy_0",
                "image": torch.rand(3, 256, 256),
                "boxes": torch.tensor([[100.0, 100.0, 50.0, 50.0]]),  # x2 < x1, degenerate!
                "labels": torch.tensor([2]),
                "ignore_mask": torch.zeros(256, 256, dtype=torch.uint8),
                "orig_size": (256, 256),
                "scale_factor": (1.0, 1.0),
            }

    results = validator.validate_dataset(DummyDataset())
    assert results["valid"] is False
    assert results["degenerate_boxes"] == 1

