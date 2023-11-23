from enum import Enum, auto
from collections.abc import Sequence
# import torch
from torch.utils.data import Dataset

# import numpy as np

from prepareable_dataset import PrepareableDataset


class DatasetFrequency(Enum):
    """The relative frequency of a dataset in a collection of multiple ones."""

    ALL_EQUAL = auto()
    PROPORTIONAL_TO_SAMPLES = auto()
    ALL_TYPES_EQUAL_PROPORTIONAL_TO_SAMPLES = auto()


class ShuffleType(Enum):
    """The relative frequency of a dataset in a collection of multiple ones."""

    DISABLED = (
        auto()
    )  # Never shuffle, go sequentially through dataset list and content each
    WITHIN_DATASET = (
        auto()
    )  # shuffle within dataset, but go sequentially through dataset list
    ACROSS_DATASETS = auto()  # shuffle within datasets and across datasets


class MultiDatasetDataet(Dataset):
    def __init__(self, datasets: list[Dataset], frequency: DatasetFrequency,
                 shuffle: ShuffleType,
                 split: Sequence[float] = [0.8, 0.1, 0.1]):

        assert sum(split) == 1.0, "Split must sum to 1.0"
        assert len(split) == 3, "Split must be a list of length 3"

        super().__init__()
        self.unprocessed_datasets = datasets
        self.frequency = frequency
        self.shuffle = shuffle
        self.split = split
        self.dataset = None

    def prepare(self):
        for dataset in self.unprocessed_datasets:
            if isinstance(dataset, PrepareableDataset):
                dataset.prepare()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return super().__getitem__(index)
