"""Class for concatinating multiple datasets."""

import math
from enum import Enum, auto
from collections.abc import Sequence
from typing import Any

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class DatasetFrequency(Enum):
    """The relative frequency of a dataset in a collection of multiple ones."""

    ALL_EQUAL = auto()
    PROPORTIONAL_TO_SAMPLES = auto()
    ALL_TYPES_EQUAL_PROPORTIONAL_TO_SAMPLES = auto()


class ShuffleType(Enum):
    """The relative frequency of a dataset in a collection of multiple ones."""

    # Never shuffle, go sequentially through dataset list and content each
    DISABLED = auto()
    # shuffle within dataset, but go sequentially through dataset list
    WITHIN_DATASET = auto()
    # shuffle within datasets and across datasets
    ACROSS_DATASETS = auto()


class ConcatDataset(Dataset):
    """Class for concatenating multiple datasets into one long one."""

    def __init__(self, datasets: list[Dataset], fractions: float | Sequence[float] = 1.0) -> None:
        """
        Initialize a new instance of the ConcatDataset class.

        Args:
            datasets (list[Dataset]): The datasets to concatenate.
            fractions (float | Sequence[float], optional): The fraction of each dataset to use.
                Must be between 0 and 1. If it is a float, the same fraction is used for all datasets.
                If it is a sequence of floats, each dataset can have a different fraction.
                Defaults to 1.0.

        Raises
        ------
            ValueError: If the number of datasets is zero.
            ValueError: If the number of datasets does not match the number of fractions.

        """
        if not datasets:
            raise ValueError("The number of datasets must be greater than zero")
        if isinstance(fractions, float):
            fractions = [fractions] * len(datasets)
        else:
            if len(datasets) != len(fractions):
                raise ValueError(
                    "The number of datasets must match the number of percentages "
                    f" but was {len(datasets)} and {len(fractions)} respectively"
                )

        self.datasets = datasets
        self.fractions = fractions

        # The number of data points in each dataset
        self.lengths = [math.ceil(len(d) * p) for d, p in zip(self.datasets, self.fractions)]
        self.total_length = sum(self.lengths)

        self.start_indices = [0]
        self.cumulative_lengths = np.cumsum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index: int) -> tuple[Any, Tensor]:
        """Get an item from the concatenated dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns
        -------
            tuple[Any, Tensor]: The item from the dataset at the given index.
        """
        dataset_index = np.searchsorted(self.cumulative_lengths, index, side="right")

        if dataset_index > 0:
            local_index = index - self.cumulative_lengths[dataset_index - 1]
        else:
            local_index = index

        return self.datasets[dataset_index][local_index]

    def __len__(self) -> int:
        """Get the total length of the concatenated dataset.

        Returns
        -------
            int: The total length of the concatenated dataset.
        """
        return self.total_length
