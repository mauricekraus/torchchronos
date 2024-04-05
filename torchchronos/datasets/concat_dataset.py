"""Class for concatinating multiple datasets."""

import math
from enum import Enum, auto
from collections.abc import Sequence
from typing import Any

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, Subset
from .prepareable_dataset import PrepareableDataset


class ShuffleMode(Enum):
    DISABLED = auto()
    WITHIN_DATASET = auto()
    ACROSS_DATASETS = auto()


class FrequencyMode(Enum):
    ALL_EQUAL = auto()
    PROPORTIONAL_TO_SAMPLE = auto()


class ConcatDataset(PrepareableDataset):

    def __init__(
        self,
        datasets: list[PrepareableDataset],
        frequency: float | Sequence[float] | FrequencyMode = FrequencyMode.PROPORTIONAL_TO_SAMPLE,
        shuffle: ShuffleMode = ShuffleMode.DISABLED,
    ) -> None:

        if not datasets:
            raise ValueError("The number of datasets must be greater than zero")
        if frequency is FrequencyMode.PROPORTIONAL_TO_SAMPLE:
            frequency = [1.0] * len(datasets)
        elif isinstance(frequency, float):
            assert len(datasets) == 1
            frequency = [frequency]
        elif isinstance(frequency, list) and len(datasets) != len(frequency):
            raise ValueError(
                "The number of datasets must match the number of percentages "
                f" but was {len(datasets)} and {len(frequency)} respectively"
            )

        self.datasets = datasets
        self.frequency = frequency
        self.shuffle = shuffle

        super().__init__()

    def _prepare(self) -> None:
        for dataset in self.datasets:
            if isinstance(dataset, PrepareableDataset):
                dataset.prepare()

    def _load(self) -> None:
        def _build_indicies(dataset: Dataset, fraction: float) -> Dataset:
            if fraction == 1.0:
                return np.arange(len(dataset))
            else:
                indicies = None
                while fraction >= 1:
                    if indicies is None:
                        indicies = np.arange(len(dataset))
                    else:
                        indicies = np.concatenate((indicies, np.arange(len(dataset))))
                    fraction -= 1

                part_indicies = np.random.permutation(len(dataset))[: math.ceil(len(dataset) * fraction)]
                indicies = part_indicies if indicies is None else np.concatenate((indicies, part_indicies))
                if self.shuffle == ShuffleMode.WITHIN_DATASET:
                    indicies = np.random.permutation(indicies)
                return indicies

        for dataset in self.datasets:
            if isinstance(dataset, PrepareableDataset):
                dataset.load()

        if self.frequency == FrequencyMode.ALL_EQUAL:
            longest_dataset = max(len(dataset) for dataset in self.datasets)
            self.frequency = [longest_dataset / len(dataset) for dataset in self.datasets]

        self.datasets = [
            Subset(dataset, _build_indicies(dataset, fraction))
            for dataset, fraction in zip(self.datasets, self.frequency)
        ]

        self.total_length = sum(len(dataset) for dataset in self.datasets)
        self.cumulative_lengths = np.cumsum([len(dataset) for dataset in self.datasets])
        self.indices = np.arange(self.total_length)
        if self.shuffle == ShuffleMode.ACROSS_DATASETS:
            np.random.shuffle(self.indices)

    def _get_item(self, index: int) -> tuple[Any, Tensor]:
        index = self.indices[index]
        dataset_index = np.searchsorted(self.cumulative_lengths, index, side="right")

        if dataset_index > 0:
            local_index = index - self.cumulative_lengths[dataset_index - 1]
        else:
            local_index = index

        return self.datasets[dataset_index][local_index]

    def __len__(self) -> int:
        return self.total_length
