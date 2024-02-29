import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

"""
This class is for concatenating multiple datasets into one long one.
There are two ways to use this class:
    1. Concatenate multiple datasets into one long one.
    2. Concatenate multiple datasets into one long one, but only use a fraction of each dataset.


"""


class ConcatDataset(Dataset):
    def __init__(
        self, datasets: list[Dataset], fractions: float | Sequence[float] = 1.0
    ) -> None:
        """A dataset that concatenates multiple datasets into one long one.

        Args:
            datasets: The datasets to concatenate
            fractions: The fraction of each dataset to use. Must be between 0 and 1.
                If it is an int, then the same fraction is used for all datasets.
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
        self.lengths = [
            math.ceil(len(d) * p) for d, p in zip(self.datasets, self.fractions)
        ]
        self.total_length = sum(self.lengths)

        self.start_indices = [0]
        self.cumulative_lengths = np.cumsum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index: int) -> tuple[Any, Tensor]:
        dataset_index = np.searchsorted(self.cumulative_lengths, index, side="right")

        if dataset_index > 0:
            local_index = index - self.cumulative_lengths[dataset_index - 1]
        else:
            local_index = index

        return self.datasets[dataset_index][local_index]

    def __len__(self) -> int:
        return self.total_length
