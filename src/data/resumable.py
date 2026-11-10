import random
from typing import Any, Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def _sample_seed(seed: int, epoch: int, position: int) -> int:
    return (seed * 1000003 + epoch * 100003 + position * 7919) & 0x7FFFFFFF


class ResumableEpochSampler(Sampler):
    """Deterministic per-epoch shuffle that can start part-way through an epoch.

    The order depends only on (seed, epoch), so skipping the first `start` positions reproduces exactly the
    remaining batches of an interrupted epoch without loading the skipped samples. Each position also gets a
    fixed augmentation seed. Yields (dataset_index, sample_seed) pairs for `SeededDataset`.
    """

    def __init__(self, size: int, seed: int = 42, shuffle: bool = True):
        self.size = size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch = epoch
        self.start = start

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        if self.shuffle:
            gen = torch.Generator().manual_seed(_sample_seed(self.seed, self.epoch, 0))
            order = torch.randperm(self.size, generator=gen).tolist()
        else:
            order = list(range(self.size))
        for position in range(self.start, self.size):
            yield order[position], _sample_seed(self.seed, self.epoch, position + 1)

    def __len__(self) -> int:
        return self.size - self.start


class SeededDataset(Dataset):
    """Runs each sample's augmentation under its own seed, leaving the caller's random state untouched."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: Tuple[int, int]) -> Any:
        index, seed = item
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            return self.dataset[index]
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
