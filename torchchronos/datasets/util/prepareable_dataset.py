from abc import ABC, abstractmethod
from typing import Optional, Any

import torch
from torch.utils.data import Dataset
import numpy as np

from ...transforms.base import Transform
from ...transforms.transforms import Identity

"""
This class is for datasets that need to be prepared before they can be used.
The idea is that the dataset is only prepared, when it is needed.
Therefor programming can be continuted and the dataset is only prepared and loaded when it is needed.
It has 2 special methods: prepare and load:
    Prepare:
        - has to be called before the load method.
        - is for e.g. downloading the data, making initial calculations, etc.
    Load:
        - is for loading the data into memory.

To use this class, you have to inherit from it and implement the following methods:
    - _prepare
    - _load
    - __getitem__
    - __len__
_prepare, and _load are called from load and prepare in the Superclass and should not be called directly.
This is because in the Superclass checks are done to make sure that the dataset is not prepared or loaded twice.
"""


class PrepareableDataset(ABC, Dataset):
    def __init__(
        self,
        transform: Transform = Identity(),
        domain: str | None = None,
    ) -> None:
        super().__init__()
        self.is_prepared: bool = False
        self.is_loaded: bool = False
        self._transform = transform
        self.domain = domain
        self.data: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None

    @property
    def transform(self) -> Transform:
        return self._transform

    def __getitem__(self, idx: int) -> Any:
        if self.is_loaded is False:
            raise NotLoadedError("Dataset must be loaded before it can be used.")
        
        time_series, target = self._get_item(idx)
        transformed_time_series, target = self._transform.transform(time_series, target)

        if target is None:
            return transformed_time_series
        
        return transformed_time_series, target


    @abstractmethod
    def _get_item(self, idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def prepare(self) -> None:
        if self.is_prepared:
            return
        self._prepare()
        self.is_prepared = True

    @abstractmethod
    def _prepare(self) -> None:
        pass

    def load(self) -> None:
        if self.is_prepared is False:
            raise NotPreparedError("Dataset must be prepared before it can be loaded.")
        self._load()
        self.is_loaded = True

        self.transform.fit(self.data, self.targets) #TODO: with whole dataset or only parts?

    @abstractmethod
    def _load(self) -> None:
        pass


class NotLoadedError(Exception):
    pass


class NotPreparedError(Exception):
    pass
