import torch
import numpy as np
from torch.utils.data import Dataset
from enum import Enum, auto

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

    def __init__(self, datasets: list[Dataset], frequency: DatasetFrequency, shuffle: ShuffleType):
        self.unprocessed_datasets = datasets
        self.frequency = frequency
        self.shuffle = shuffle
        self._create_dataset()
        

    def _create_dataset(self):
        if self.shuffle == ShuffleType.ACROSS_DATASETS:
            self.dataset = np.concatenate(self.unprocessed_datasets)
            np.random.shuffle(self.dataset)
        elif self.shuffle == ShuffleType.WITHIN_DATASET:
            for dataset in self.unprocessed_datasets:
                np.random.shuffle(dataset) # inplace??
            self.dataset = np.concatenate(self.unprocessed_datasets)
        else:
            self.dataset = np.concatenate(self.unprocessed_datasets)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    



    