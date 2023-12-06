from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

import numpy as np
from aeon.datasets._data_loaders import download_dataset, load_classification


class PrepareableDataset(ABC, Dataset):
    def __init__(self, prepare=False, load=False) -> None:
        super().__init__()
        self.is_prepared = False
        self.is_loaded = False
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
        self.is_loaded = True

    def _load(self):
        pass


class AeonDataset(PrepareableDataset):
    def __init__(self, name: str, split=None):
        super().__init__()
        self.X = None
        self.y = None
        self.meta_data = None
        self.name = name
        self.split = split

    def __len__(self):
        if not self.is_loaded:
            raise Exception("Dataset must be loaded before it can be used.")
        return len(self.X)

    def __getitem__(self, idx):
        if not self.is_loaded:
            raise Exception("Dataset must be loaded before it can be used.")
        return self.X[idx], self.y[idx]

    def _prepare(self):
        # replace with download_dataset
        download_dataset(self.name, save_path=".cache/data/")

    def _load(self):
        self.X, self.y, self.meta_data = load_classification(
            self.name, split=self.split, extract_path=".cache/data/"
        )
        self._update_labels()
        self.X = self.X.astype("float32")

    def _update_labels(self):
        labels = []
        for label in self.y:
            l = int(label)
            if l not in labels:
                labels.append(l)
        labels = np.sort(labels)
        for idx, label in enumerate(labels):
            self.y[self.y == str(label)] = idx