import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import overload, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..datasets.util.base_dataset import BaseDataset

class PreparableDataset:
    pass

# TODO: implement  Reshape Transform, MinMax Transform


class Transform(ABC):
    def __init__(self, is_fitted: bool = False):
        self.is_fitted = is_fitted
        self._invert_transform: Optional["Transform"] = None

    @overload
    def __call__(self, time_series: torch.Tensor) -> torch.Tensor:
        ...

    @overload
    def __call__(self, time_series: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...
    
    @overload 
    def __call__(self, time_series: Dataset, targets: None) -> Dataset:
        ...

    def __call__(
        self, time_series: Dataset | torch.Tensor, targets: Optional[torch.Tensor]=None
    ) -> Dataset | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if isinstance(time_series, Dataset):
            time_series = self.transform(time_series)
            return time_series
        if targets is None:
            time_series = self.transform(time_series)
            return time_series
        return self.transform(time_series, targets)

    def __add__(self, other:"Transform") -> "Compose":
        return Compose([self, other])

    def __invert__(self) -> "Transform":
        if self._invert_transform is None:
            self._invert_transform = self._invert()
            self._invert_transform._invert_transform = self
            self._invert_transform.is_fitted = True
        return self._invert_transform

    def save(self, name: str, path: Optional[Path] = None) -> None:
        if not self.is_fitted:
            raise Exception("Transform must be fitted before it can be saved.")

        if path is None:
            path = Path(".cache/torchchronos/transforms")

        path.mkdir(parents=True, exist_ok=True)
        file_path = path / (name + ".pkl")

        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(name: str, path: Optional[Path] = None):
        if path is None:
            path = Path(".cache/torchchronos/transforms")

        file_path = path / (name + ".pkl")

        with open(file_path, "rb") as file:
            transform = pickle.load(file)
        return transform

    @overload
    def fit_transform(self, time_series: torch.Tensor) -> torch.Tensor:
        ...
    
    @overload
    def fit_transform(self, time_series: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def fit_transform(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]=None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        
        self.fit(time_series, targets)
        if targets is None:
            time_series = self.transform(time_series)
            self.is_fitted = True
            return time_series
        else:
            time_series, target = self.transform(time_series, targets)
            self.is_fitted = True
            return time_series, target

    #TODO: Datasets can not be fit, with this implementation
    def fit(self, time_series:torch.Tensor, targets:Optional[torch.Tensor]=None) -> None:
        if self.is_fitted:
            return
        self._fit(time_series, targets)
        self.is_fitted = True

    @overload
    def transform(self, time_series: torch.Tensor, targets: None = None) -> torch.Tensor:
        ...

    @overload
    def transform(self, time_series: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def transform(self, time_series: Dataset, targets: None) -> Dataset:
        ... 

    def transform(self, time_series: Dataset | torch.Tensor, targets:Optional[torch.Tensor]=None) -> Dataset | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.is_fitted is False:
            raise Exception("Transform must be fitted before it can be used.")

        if isinstance(time_series, Dataset):
            if targets is not None:
                raise RuntimeError("If tranforming a dataset, targets have to be None. Do not provide targets to the transform method!")
            dataset_transformed = self._transform_dataset(time_series)
            return dataset_transformed
        if targets is None:
            ts_transformed, _ = self._transform(time_series, None)
            return ts_transformed
        else:
            idk = self._transform(time_series, targets)
            ts_transformed, targets_transformed = idk

            # Mainly for the RemoveLabels Transform
            if targets_transformed is None:
                return ts_transformed
            else:
                return ts_transformed, targets_transformed


    def _transform_dataset(self, dataset: Dataset) -> BaseDataset:
        try:
            data = dataset[:]
            if isinstance(data, tuple) and len(data) == 2:
                data, targets = data
            else:
                targets = None
        except:
            #TODO: implement for collecting the data with iterating over the dataset
            pass
        ts_transformed, targets_transformed = self._transform(data, targets)

        return BaseDataset(ts_transformed, targets_transformed)
        
        
    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    @overload
    @abstractmethod
    def _transform(self, time_series: torch.Tensor, targets: None = None) -> tuple[torch.Tensor, None]:
        ...

    @overload
    @abstractmethod
    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...


    @abstractmethod
    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, None]:
        pass

    @abstractmethod
    def _invert(self):
        pass


class Compose(Transform):
    def __init__(self, transforms: list[Transform], is_fitted:bool=False):
        super().__init__(is_fitted=is_fitted)
        self.transforms:list[Transform] = transforms


    def __add__(self, other:Transform) -> "Compose":
        new_compose = Compose([*self.transforms, other])
        return new_compose

    def __getitem__(self, index:int) -> Transform:
        return self.transforms[index]


    def fit_transform(
        self, time_series, targets
    ):
        self.fit(time_series, targets)
        if targets is None:
            time_series = self.transform(time_series)
            self.is_fitted = True
            return time_series
        else:
            time_series, target = self.transform(time_series, targets)
            self.is_fitted = True
            return time_series, target

    def _fit(
            self, time_series:torch.Tensor, targets:Optional[torch.Tensor]=None
            ) -> None:
        if self.transforms == []:
            return 

        if targets is None:
            for t in self.transforms[:-1]:
                time_series = t.fit_transform(time_series)

            self.transforms[-1].fit(time_series)
        else:
            for t in self.transforms[:-1]:
                time_series, targets = t.fit_transform(time_series, targets)

            self.transforms[-1].fit(time_series, targets)
        self.is_fitted = True


    def _transform(self, time_series, targets = None):
        
        for t in self.transforms:
            ts_transformed = t.transform(time_series, targets)
            if isinstance(ts_transformed, tuple):
                time_series, targets = ts_transformed
            else:
                time_series, targets = ts_transformed, None
        
        return time_series, targets

    def _invert(self) -> Transform:
        return Compose([~t for t in self.transforms[::-1]])

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string



