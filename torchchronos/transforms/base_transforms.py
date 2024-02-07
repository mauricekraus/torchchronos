import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import overload, Optional

import numpy as np
import torch

# TODO: implement Dtype Transform, Reshape Transform, change from torch.tensor to torch.Tensor


class Transform(ABC):
    def __init__(self, is_fitted: bool = False):
        self.is_fitted = is_fitted
        self._invert_transform: Optional["Transform"] = None

    @overload
    def __call__(self, time_series: torch.Tensor) -> torch.Tensor:
        pass

    @overload
    def __call__(
        self, time_series: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def __call__(
        self, time_series: torch.Tensor, targets: Optional[torch.Tensor]=None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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

    def fit(self, time_series:torch.Tensor, targets:Optional[torch.Tensor]=None) -> None:
        if self.is_fitted:
            return
        self._fit(time_series, targets)
        self.is_fitted = True

    @overload
    def transform(self, time_series: torch.Tensor) -> torch.Tensor:
        ...

    @overload
    def transform(
        self, time_series: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor]=None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.is_fitted is False:
            raise Exception("Transform must be fitted before it can be used.")

        if targets is None:
            ts_transformed, _ = self._transform(time_series)
            return ts_transformed
        else:
            ts_transformed, labels_transformed = self._transform(time_series, targets)
            return ts_transformed, labels_transformed

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
    def _transform(self, time_series: torch.Tensor) -> tuple[torch.Tensor, None]:
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

    @overload
    def _transform(self, time_series: torch.Tensor) -> tuple[torch.Tensor, None]:
        ...

    @overload
    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, None]:
        if targets is None:
            for t in self.transforms:
                time_series = t.transform(time_series)
            return time_series, None
        else:
            for t in self.transforms:
                time_series, targets = t.transform(time_series, targets)
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
