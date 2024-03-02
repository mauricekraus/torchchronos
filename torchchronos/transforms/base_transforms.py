import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import overload, Optional

import torch
from torch.utils.data import Dataset

from ..datasets.base_dataset import BaseDataset


# TODO: implement Reshape Transform, MinMax Transform


def get_data_from_dataset(dataset: Dataset) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    data = dataset[:]
    if isinstance(data, tuple) and len(data) == 2:
        data, targets = data
    else:
        targets = None
    return data, targets


class Transform(ABC):
    def __init__(self, is_fitted: bool = False):
        self.is_fitted = is_fitted
        self._invert_transform: Optional["Transform"] = None

    @overload
    def __call__(self, time_series: torch.Tensor) -> torch.Tensor: ...

    @overload
    def __call__(
        self, time_series: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(self, time_series: Dataset) -> Dataset: ...

    def __call__(
        self, time_series: Dataset | torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> BaseDataset | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if targets is None:
            ts_transformed = self.transform(time_series)
            return ts_transformed
        else:
            assert not isinstance(time_series, Dataset)
            return self.transform(time_series, targets)

    def __add__(self, other: "Transform") -> "Compose":
        return Compose([self, other])

    def __invert__(self) -> "Transform":
        return self.invert()

    def invert(self) -> "Transform":

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
    def fit_transform(self, time_series: torch.Tensor) -> torch.Tensor: ...

    @overload
    def fit_transform(
        self, time_series: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def fit_transform(self, time_series: Dataset) -> BaseDataset: ...

    def fit_transform(
        self, time_series: Dataset | torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> BaseDataset | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        self.fit(time_series, targets)
        if targets is None:
            ts_transformed = self.transform(time_series)
            self.is_fitted = True
            return ts_transformed
        else:
            assert not isinstance(time_series, Dataset)
            ts_transformed, targets_transformed = self.transform(time_series, targets)
            self.is_fitted = True
            return ts_transformed, targets_transformed

    def fit(self, time_series: Dataset | torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        if self.is_fitted:
            return
        if isinstance(time_series, Dataset):
            assert targets is None
            self._fit_dataset(time_series)
        else:
            self._fit(time_series, targets)
        self.is_fitted = True

    @overload
    def transform(self, time_series: torch.Tensor) -> torch.Tensor: ...

    @overload
    def transform(
        self, time_series: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def transform(self, time_series: Dataset) -> BaseDataset: ...

    def transform(
        self, time_series: Dataset | torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> BaseDataset | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.is_fitted is False:
            raise Exception("Transform must be fitted before it can be used.")

        if isinstance(time_series, Dataset):
            if targets is not None:
                raise RuntimeError(
                    "If tranforming a dataset, targets have to be None. Do not provide targets to the transform method!"
                )
            dataset_transformed = self._transform_dataset(time_series)
            return dataset_transformed
        if targets is None:
            time_series, _ = self._transform(time_series)
            return time_series
        return self._transform(time_series, targets)

    def _transform_dataset(self, dataset: Dataset) -> BaseDataset:
        data, targets = get_data_from_dataset(dataset)
        # TODO: Use TensorDataset instead of BaseDataset
        if targets is None:

            ts_transformed, _ = self._transform(data)
            return BaseDataset(ts_transformed)

        ts_transformed, targets_transformed = self._transform(data, targets)
        return BaseDataset(ts_transformed, targets_transformed)

    def _fit_dataset(self, dataset: Dataset) -> None:
        data, targets = get_data_from_dataset(dataset)
        if targets is None:
            self._fit(data)
        else:
            self._fit(data, targets)

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def _fit(self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        pass

    @abstractmethod
    def _transform(
        self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

    @abstractmethod
    def _invert(self):
        pass


class Compose(Transform):
    def __init__(self, transforms: list[Transform]):
        all_fitted = all([t.is_fitted for t in transforms])
        super().__init__(is_fitted=all_fitted)
        self.transforms: list[Transform] = transforms

    def __add__(self, other: Transform) -> "Compose":
        new_compose = Compose([*self.transforms, other])
        return new_compose

    def __getitem__(self, index: int) -> Transform:
        return self.transforms[index]

    def _fit(self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
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

    def _transform(
        self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        if targets is None:
            for t in self.transforms:
                time_series = t.transform(time_series)
            return time_series, None
        elif isinstance(targets, torch.Tensor):
            for t in self.transforms:
                ts_transformed = t.transform(time_series, targets)
                time_series, targets = ts_transformed

            return time_series, targets
        else:
            raise TypeError("Targets must be a tensor or None")

    def _invert(self) -> Transform:
        return Compose([~t for t in self.transforms[::-1]])

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
