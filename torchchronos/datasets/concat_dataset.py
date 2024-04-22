"""Module for concatinating multiple datasets.

This module implements the functionality to combine mutliple datasets into one. The apart fromt the
in the ConcatDataset class, there are two Enums: ShuffleMode and FrequencyMode. The ShuffleMode describes
how the indices are shuffled, while the FrequencyMode describes how the datasets can be are combined.
"""

import math
from collections.abc import Sequence
from enum import Enum, auto
from typing import Any

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, Subset

from torchchronos.transforms import Identity, Transform

from .prepareable_dataset import PrepareableDataset


class ShuffleMode(Enum):
    """The different shuffle modes for the ConcatDataset.

    Args:
        DISABLED: No shuffling is done.
        WITHIN_DATASET: The indices are shuffled within each dataset.
        ACROSS_DATASETS: The indices are shuffled across all datasets.
    """

    DISABLED = auto()
    WITHIN_DATASET = auto()
    ACROSS_DATASETS = auto()


class FrequencyMode(Enum):
    """The different frequency modes for the ConcatDataset.

    Keep in mind this is only one way of specify the way to combine the datasets.
    ALL_EQUAL: All datasets are used the same number of times.
    PROPORTIONAL_TO_SAMPLE: The datasets are used according to the given fractions.
    TODO: Add sampling frequency for types.
    """

    ALL_EQUAL = auto()
    PROPORTIONAL_TO_SAMPLE = auto()


class ConcatDataset(PrepareableDataset):
    """Dataset concating multipe datasets together.

    This class allows to chain multiple datasets togehter. It is possilbe to combine normal torch.Datasets
    with PrepareableDatests. It is a PrepareableDataset itself, and will call prepare and load on all
    PrepareableDatasets in the datasets list. There are different ways to combine the datasets. It is
    possible to use one of the FrequencyModes to specify how often each dataset should be used,
    or specify a list of floats with the fractions of the datasets to be used.When using a frequenca >= 1,
    will use the dataset as many times as the integer part of the frequency and sample the rest randomly
    acroding to the proportion of the fraction. The ShuffleMode can be used to shuffle the indices within
    each dataset or across all datasets.

    Args:
        datasets: The datasets to be concatenated.
        frequency: The frequency of the datasets. Can be a float, a list of floats or a FrequencyMode.
                    This describes how much of the according datset is in the new concated dataset.
        shuffle: The shuffle mode of the dataset.

    Examples:
        This example shows how to create a dataset containng 3 times the same dataset.

        >>> from torchchronos.datasets import ConcatDataset, AeonClassificationDataset
        >>> gun_point = AeonClassificationDataset("GunPoint")
        >>> dataset = ConcatDataset([gun_point], [3.0])
        >>> dataset.prepare()
        >>> dataset.load()

        This example shows how to create a dataset containing 3 different datasets. The the amount of each
        dataset is proportional to the given fractions.

        >>> from torchchronos.datasets import ConcatDataset, AeonClassificationDataset
        >>>
        >>> gun_point = AeonClassificationDataset("GunPoint")
        >>> arrow_head = AeonClassificationDataset("ArrowHead")
        >>> coffee = AeonClassificationDataset("Coffee")
        >>> dataset = ConcatDataset([gun_point, arrow_head, coffee], [0.5, 0.3, 0.2])
        >>> dataset.prepare()
        >>> dataset.load()

        In the last example each time series of the dataset has a different length. This could run into
        problems when batches are created.To avoid this, either add a transform to the ConcatDataset or
        add transformations to the datasets before concatenating them.

        >>> from torchchronos.datasets import ConcatDataset, AeonClassificationDataset
        >>> from torchchronos.transforms import Crop
        >>>
        >>> crop = Crop(0, 100)
        >>>
        >>> gun_point = AeonClassificationDataset("GunPoint", transform=crop)
        >>> arrow_head = AeonClassificationDataset("ArrowHead", transform=crop)
        >>> coffee = AeonClassificationDataset("Coffee", transform=crop)
        >>> dataset = ConcatDataset([gun_point, arrow_head, coffee], [0.5, 0.3, 0.2])
        >>> # or
        >>> gun_point = AeonClassificationDataset("GunPoint")
        >>> arrow_head = AeonClassificationDataset("ArrowHead")
        >>> coffee = AeonClassificationDataset("Coffee")
        >>> dataset = ConcatDataset([gun_point, arrow_head, coffee], [0.5, 0.3, 0.2], transform=crop)
        >>>
        >>> dataset.prepare()
        >>> dataset.load()

    """

    def __init__(
        self,
        datasets: list[PrepareableDataset],
        frequency: (float | Sequence[float] | FrequencyMode) = FrequencyMode.PROPORTIONAL_TO_SAMPLE,
        shuffle: ShuffleMode = ShuffleMode.DISABLED,
        transform: Transform = Identity(),
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
        self.transform = transform

        super().__init__()

    def _prepare(self) -> None:
        """Calls prepare on all PrepareableDatasets in the datasets list."""
        for dataset in self.datasets:
            if isinstance(dataset, PrepareableDataset):
                dataset.prepare()

    def _load(self) -> None:
        """Loads all PrepareableDatasets in the datasets list and builds the indices.

        This method will call load on all PrepareableDatasets in the datasets list. After that it will use
        the now availavle length attirbute to determine the indices of the new dataset. If the FreqencyMode
        is ALL_EQUAL, the longest dataset will be used as the reference length.
        """
        rnd = np.random.default_rng()

        def _build_indicies(dataset: Dataset, fraction: float) -> Dataset:
            if fraction == 1.0:
                return np.arange(len(dataset))
            else:
                indicies = None  # np.empty((0, )) und dann kein if/else
                while fraction >= 1:
                    if indicies is None:
                        indicies = np.arange(len(dataset))
                    else:
                        indicies = np.concatenate((indicies, np.arange(len(dataset))))
                    fraction -= 1

                part_indicies = rnd.permutation(len(dataset))[: math.ceil(len(dataset) * fraction)]
                indicies = part_indicies if indicies is None else np.concatenate((indicies, part_indicies))
                if self.shuffle == ShuffleMode.WITHIN_DATASET:
                    indicies = rnd.permutation(indicies)
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
            rnd.shuffle(self.indices)  # This is in place

    def _get_item(self, index: int) -> tuple[Any, Tensor]:
        """Returns the item at the given index.

        The dataset might be shuffled, therefore the index is first looked up in the indecies list.
        A Search is done to first determine the right dataset and then the local index in the dataset.

        Args:
            index: The index of the item.

        """
        index = self.indices[index]
        dataset_index = np.searchsorted(self.cumulative_lengths, index, side="right")

        if dataset_index > 0:
            local_index = index - self.cumulative_lengths[dataset_index - 1]
        else:
            local_index = index

        return self.datasets[dataset_index][local_index]

    def __len__(self) -> int:
        return self.total_length
