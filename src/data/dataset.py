import abc
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, abc.ABC):
    """Abstract base dataset enforcing a common internal representation across all datasets.
    
    Output dictionary schema per sample:
        image_id (str): Unique image identifier.
        image (torch.Tensor): Float32 tensor [3, H, W] normalized to [0.0, 1.0].
        boxes (torch.Tensor): Float32 tensor [N, 4] in absolute pixels [x1, y1, x2, y2].
        labels (torch.Tensor): Int64 tensor [N] with class IDs in [0, num_classes - 1].
        ignore_mask (torch.Tensor): UInt8 tensor [H, W] where 1 indicates ignore region, 0 valid.
        orig_size (Tuple[int, int]): (original_height, original_width).
        scale_factor (Tuple[float, float]): (scale_x, scale_y) applied during resizing.
    """

    def __init__(self, root_dir: str, split: str, img_size: Tuple[int, int] = (1024, 1024), is_train: bool = False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.is_train = is_train

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass

