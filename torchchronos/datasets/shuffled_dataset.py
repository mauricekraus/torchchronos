from collections.abc import Sequence

import torch
from torch.utils.data import Dataset, Subset

"""
This class is to shuffle a dataset.
It wrapps the Pytorch.Subset class, but it shuffles the indices before passing them to the Subset class.
"""


class ShuffledDataset(Subset):
    def __init__(self, dataset: Dataset) -> None:
        indices: Sequence[int] = torch.randperm(len(dataset))
        super().__init__(dataset, indices)
