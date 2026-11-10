import os
import random
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.resumable import ResumableEpochSampler, SeededDataset
from src.utils.checkpoint import create_checkpoint_payload, load_checkpoint, save_atomic


class RandomAugmentDataset(Dataset):
    def __len__(self):
        return 23

    def __getitem__(self, idx):
        return idx, random.random()


def batches(start_batch, epoch, batch_size=4):
    sampler = ResumableEpochSampler(23, seed=7)
    sampler.set_epoch(epoch, start=start_batch * batch_size)
    loader = DataLoader(SeededDataset(RandomAugmentDataset()), batch_size=batch_size, sampler=sampler, drop_last=True,
                        collate_fn=lambda b: b)
    return list(loader)


def test_resumed_epoch_matches_uninterrupted_tail():
    full = batches(0, epoch=3)
    resumed = batches(2, epoch=3)
    assert len(full) == 23 // 4
    assert resumed == full[2:], "same samples and same augmentation draws after resume"


def test_epochs_shuffle_differently_and_caller_rng_is_untouched():
    random.seed(123)
    expected = random.random()
    random.seed(123)
    a, b = batches(0, epoch=1), batches(0, epoch=2)
    assert random.random() == expected
    assert [x for batch in a for x, _ in batch] != [x for batch in b for x, _ in batch]


def test_mid_epoch_checkpoint_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        resume = {"epoch": 7, "batch": 1200, "batch_size": 2, "epoch_losses": {"loss_total": 3.5}}
        path = os.path.join(tmp, "latest.pth")
        save_atomic(create_checkpoint_payload(model, optimizer, None, None, 6, 20612, 1, 0.1, {"a": 1}, resume), path)

        info = load_checkpoint(path, model, optimizer)
        assert (info["start_epoch"], info["start_batch"]) == (7, 1200)
        assert info["epoch_losses"] == {"loss_total": 3.5}
        assert info["resume_batch_size"] == 2
        assert info["global_step"] == 20612


def test_end_of_epoch_checkpoint_starts_next_epoch():
    with tempfile.TemporaryDirectory() as tmp:
        model = nn.Linear(4, 2)
        path = os.path.join(tmp, "latest.pth")
        save_atomic(create_checkpoint_payload(model, torch.optim.SGD(model.parameters(), 0.1), None, None, 6, 0, 1, 0.0, {}), path)
        info = load_checkpoint(path, model)
        assert (info["start_epoch"], info["start_batch"]) == (7, 0)
