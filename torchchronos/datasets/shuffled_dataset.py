import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset


class ShuffledDataset(Subset):
    def __init__(self, dataset: Dataset) -> None:
        indices: Sequence[int] = torch.randperm(len(dataset))  # type: ignore
        super().__init__(dataset, indices)
