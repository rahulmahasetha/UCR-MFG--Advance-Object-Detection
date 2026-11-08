import os
import random
import tempfile
import numpy as np
import torch
import torch.nn as nn
import pytest

from src.utils.checkpoint import (
    save_atomic,
    load_checkpoint,
    create_checkpoint_payload,
    purge_old_checkpoints,
)


def test_atomic_checkpointing_and_resumption():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Mock training step
        x = torch.randn(4, 10)
        y = model(x).sum()
        y.backward()
        optimizer.step()

        # Set specific PRNG states
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
        rand_val_before = random.random()

        # Reset seed before creating payload to test restoration
        random.seed(12345)
        cfg_dict = {"model": "test", "lr": 0.01}
        payload = create_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            epoch=15,
            global_step=640,
            current_stage=2,
            best_metric=0.385,
            cfg_dict=cfg_dict
        )

        ckpt_file = os.path.join(tmp_dir, "checkpoint.pth")
        save_atomic(payload, ckpt_file)
        assert os.path.exists(ckpt_file)

        # Perturb model and RNG
        with torch.no_grad():
            model.weight.fill_(999.0)
        random.seed(99999)

        # Resume
        resumed = load_checkpoint(ckpt_file, model, optimizer, current_cfg=cfg_dict)
        assert resumed["start_epoch"] == 16  # Strictly epoch + 1
        assert resumed["global_step"] == 640
        assert resumed["current_stage"] == 2
        assert resumed["best_metric"] == 0.385

        # Check PRNG restoration
        rand_val_after = random.random()
        assert rand_val_before == rand_val_after


def test_purge_old_checkpoints():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(1, 6):
            p = os.path.join(tmp_dir, f"epoch_{i:03d}.pth")
            with open(p, "w") as f:
                f.write("test")

        # Keep last 2
        purge_old_checkpoints(tmp_dir, keep_last_n=2)
        remaining = sorted(os.listdir(tmp_dir))
        assert remaining == ["epoch_004.pth", "epoch_005.pth"]

