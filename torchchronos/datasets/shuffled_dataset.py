"""
Class to shuffle a dataset.

It wrapps the Pytorch.Subset class, but it shuffles the indices before passing them to the Subset class.
"""

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset, Subset


class ShuffledDataset(Subset):
    """A class for shuffling a dataset."""

    def __init__(self, dataset: Dataset) -> None:
        """
        Initialize a new instance of the ShuffledDataset class.

        Args:
            dataset (Dataset): The dataset to shuffle.
        """
        indices: Sequence[int] = torch.randperm(len(dataset))
        super().__init__(dataset, indices)
