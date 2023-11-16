import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


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

        #: The number of data points in each dataset
        self.lengths = [
            math.ceil(len(d) * p) for d, p in zip(self.datasets, self.fractions)
        ]
        self.total_length = sum(self.lengths)

        self.start_indices = [0]
        for i in range(len(self.datasets) - 1):
            self.start_indices.append(self.start_indices[-1] + self.lengths[i])

    def __getitem__(self, index: int) -> tuple[Any, Tensor]:
        # Determine which dataset the item belongs to
        dataset_index = next(
            i for i, si in enumerate(self.start_indices) if index < si + self.lengths[i]
        )

        # Determine the index of the item in the dataset
        item_index = index - self.start_indices[dataset_index]

        # Return the item from the selected dataset
        return self.datasets[dataset_index][item_index], torch.tensor(
            dataset_index, dtype=torch.long
        )

    def __len__(self) -> int:
        return self.total_length
