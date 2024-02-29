from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset

from ..transforms.base_transforms import Transform
from ..transforms.basic_transforms import Identity


class PrepareableDataset(ABC, Dataset):
    def __init__(
        self,
        transform: Transform = Identity(),
        domain: str | None = None,
    ) -> None:
        self.is_prepared: bool = False
        self.is_loaded: bool = False
        self._transform: list[Transform] = transform
        self.domain = domain

    @property
    def transforms(self) -> Transform:
        return self._transform

    def __getitem__(self, idx: int) -> Any:
        if self.is_prepared is False:
            raise NotPreparedError("Dataset must be prepared before it can be used.")
        elif self.is_loaded is False:
            raise NotLoadedError("Dataset must be loaded before it can be used.")

        time_series = self._get_item(idx)
        if isinstance(time_series, tuple):
            time_series, targets = time_series
        else:
            targets = None

        if time_series.ndim == 2:
            time_series = time_series.unsqueeze(0)

        return self.transforms.transform(time_series, targets)

    @abstractmethod
    def _get_item(self, idx: int) -> Any:
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

        self.transforms.fit(self.data, self.targets)

    @abstractmethod
    def _load(self) -> None:
        pass


class NotLoadedError(Exception):
    pass


class NotPreparedError(Exception):
    pass
