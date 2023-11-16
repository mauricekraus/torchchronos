from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

from aeon.datasets import load_classification


class PrepareableDataset(ABC, Dataset):
    def __init__(self, prepare=False, load=False) -> None:
        super().__init__()
        self.is_prepared = False
        if prepare:
            self.prepare()
        if load:
            self.prepare()
            self.load()

    @abstractmethod
    def __getitem__(self, index) -> torch.Tensor:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def prepare(self):
        if self.is_prepared:
            return
        self._prepare()
        self.is_prepared = True

    def _prepare(self):
        pass

    def load(self):
        if self.is_prepared is False:
            raise Exception("Dataset must be prepared "
                            "before it can be loaded.")
        self._load()

    def _load(self):
        pass

    def __str__(self) -> str:
        return ("This is an abstract dataset class."
                "For using this class, the methods __getitem__,"
                "__len__, _prepare, _load must be implemented.")


class ClassificationDataset(PrepareableDataset):
    def __init__(self, name: str, split=None):
        super().__init__()
        self.X = None
        self.y = None
        self.meta_data = None
        self.name = name
        self.split = split

    def __len__(self):
        if self.X is None:
            raise Exception("Dataset must be loaded before it can be used.")
        return len(self.X)

    def __getitem__(self, idx):
        if self.X is None:
            raise Exception("Dataset must be loaded before it can be used.")
        return self.X[idx], self.y[idx]

    def _prepare(self):
        load_classification(self.name, split=self.split, extract_path="data/")

    def _load(self):
        self.X, self.y, self.meta_data = load_classification(
            self.name, split=self.split, extract_path="data/"
        )
        self.X = self.X.astype("float32")
