from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from aeon.datasets import load_classification


def load_wrapper(original_load):
    def wrapper(self, *args, **kwargs):
        if not self.is_prepared:
            raise Exception("Dataset must be prepared before it can be loaded.")
        elif self.is_loaded:
            return
        original_load(self, *args, **kwargs)
        self.is_loaded = True
    return wrapper


def prepare_wrapper(original_prepare):
    def wrapper(self, *args, **kwargs):
        if self.is_prepared:
            return
        original_prepare(self, *args, **kwargs)
        self.is_prepared = True
    return wrapper


def getitem_wrapper(original_getitem):
    def wrapper(self, index):
        if not self.is_loaded:
            raise Exception("Dataset must be loaded before it can be used.")
        return original_getitem(self, index)
    return wrapper


def len_wrapper(original_len):
    def wrapper(self):
        if not self.is_loaded:
            raise Exception("Dataset must be loaded before it can be used.")
        return original_len(self)
    return wrapper


class PrepareableDataset(ABC, torch.utils.data.Dataset):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
     
        # Override load method
        cls.load = load_wrapper(cls.load)

        # Override prepare method
        cls.prepare = prepare_wrapper(cls.prepare)

        # Override __getitem__ method
        cls.__getitem__ = getitem_wrapper(cls.__getitem__)

        # Override __len__ method
        cls.__len__ = len_wrapper(cls.__len__)

    def __init__(self, prepare: bool = False, load: bool = False) -> None:
        super().__init__()
        self.is_prepared: bool = False
        self.is_loaded: bool = False
        if prepare:
            self.prepare()
        if load:
            self.load()

    @abstractmethod
    def __getitem__(self, index: int) -> torch.Tensor:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def load(self):
        pass


class ClassificationDataset(PrepareableDataset):
    def __init__(self, name: str, split: str | None = None):
        super().__init__()

        self.y: List[int] | None = None
        self.meta_data: Dict | None = None
        self.name: str = name
        self.split: str | None = split

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx], self.y[idx]

    def prepare(self):
        load_classification(self.name, split=self.split, extract_path="data/")

    def load(self):
        X, self.y, self.meta_data = load_classification(
            self.name, split=self.split, extract_path="data/"
        )
        self.X = torch.from_numpy(X.astype("float32"))
