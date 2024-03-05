"""Base class for all transforms in the TorchChronos library.

This module provides the Tranform class and a class for composing multiple transforms.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import overload

import dill
import torch
from torch.utils.data import Dataset

from ..datasets.base_dataset import BaseDataset

# TODO: implement Reshape Transform, MinMax Transform


def get_data_from_dataset(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Get the data and targets from a dataset.

    Args:
        dataset (Dataset): The input dataset.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]: The data and targets from the dataset.
    """
    data = dataset[:]
    if isinstance(data, tuple) and len(data) == 2:
        data, targets = data
    else:
        targets = None
    return data, targets


class Transform(ABC):
    """
    Base class for all transforms in the TorchChronos library.

    Transforms are used to preprocess time series data before feeding it into a model.
    This class provides the basic structure and methods that all transforms should implement.

    Args:
        is_fitted (bool, optional): Indicates whether the transform has been fitted. Defaults to False.
    """

    def __init__(self, is_fitted: bool = False):
        """
        Initialize a new instance of the Transform class.

        Args:
            is_fitted (bool, optional): Indicates whether the transform is fitted or not. Defaults to False.
        """
        self.is_fitted = is_fitted
        self._invert_transform: "Transform" | None = None

    @overload
    def __call__(self, time_series: torch.Tensor) -> torch.Tensor:
        ...

    @overload
    def __call__(self, time_series: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def __call__(self, time_series: Dataset) -> Dataset:
        ...

    def __call__(
        self, time_series: Dataset | torch.Tensor, targets: torch.Tensor | None = None
    ) -> BaseDataset | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the transformation to the input time series and targets (if provided).

        Args:
            time_series (Dataset | torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target values associated with the time series data.
                                            Has to be None if time_series is a Dataset.

        Returns
        -------
            BaseDataset: If a dataset is provided.
            torch.Tensor: If only the time series is provided.
            tuple[torch.Tensor, torch.Tensor]: If both the time series and targets are provided.

        Raises
        ------
            AssertionError: If `targets` is provided but `time_series` is an instance of `Dataset`.
        """
        if targets is None:
            ts_transformed = self.transform(time_series)
            return ts_transformed
        else:
            assert not isinstance(time_series, Dataset)
            return self.transform(time_series, targets)

    def __add__(self, other: "Transform") -> "Compose":
        """
        Add a transform to the composition.

        Args:
            other (Transform): The transform to be added.

        Returns
        -------
        Compose: A new composition with the added transform.
        """
        return Compose([self, other])

    def __invert__(self) -> "Transform":
        """
        Invert the transform.

        Returns
        -------
        Transform: The inverted transform.
        """
        return self.invert()

    def invert(self) -> "Transform":
        """
        Return the inverted transform.

        If the inverted transform has not been computed yet, it is computed and stored for future use.
        The inverted transform is computed by calling the `_invert` method of the current transform object.
        The computed inverted transform is then stored in the `_invert_transform` attribute
            of the current transform object.

        Returns
        -------
            Transform: The inverted transform.

        """
        if self._invert_transform is None:
            self._invert_transform = self._invert()
            self._invert_transform._invert_transform = self
            self._invert_transform.is_fitted = True
        return self._invert_transform

    def save(self, name: str, path: Path | None = None) -> None:
        """
        Save the fitted transform object to a file.

        Args:
            name (str): The name of the saved file.
            path (Path, optional): The path where the file will be saved.
                                    Default path is ".cache/torchchronos/transforms".

        Returns
        -------
            None

        Raises
        ------
            Exception: If the transform is not fitted before saving.

        """
        if not self.is_fitted:
            raise Exception("Transform must be fitted before it can be saved.")

        if path is None:
            path = Path(".cache/torchchronos/transforms")

        path.mkdir(parents=True, exist_ok=True)
        file_path = path / (name + ".pkl")

        with open(file_path, "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load(name: str, path: Path | None = None):
        """
        Load a transform object from a pickle file.

        Args:
            name (str): The name of the transform object.
            path (Path, optional): The path to the directory containing the pickle file. Default path is used.

        Returns
        -------
            The loaded transform object.

        Raises
        ------
            FileNotFoundError: If the specified pickle file does not exist.
            pickle.UnpicklingError: If there is an error while unpickling the transform object.
        """
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
    def fit_transform(
        self, time_series: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def fit_transform(self, time_series: Dataset) -> BaseDataset:
        ...

    def fit_transform(
        self, time_series: Dataset | torch.Tensor, targets: torch.Tensor | None = None
    ) -> BaseDataset | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Fits and transforms the input time series and optional targets.

        Args:
            time_series (Dataset | torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target values associated with the time series.
                        Has to be None if time_series is a Dataset.

        Returns
        -------
            BaseDataset: If a dataset is provided.
            torch.Tensor: If only the time series is provided.
            tuple[torch.Tensor, torch.Tensor]: If both the time series and targets are provided.

        """
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

    def fit(self, time_series: Dataset | torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fits the transform to the given time series data.

        Args:
            time_series (Dataset | torch.Tensor): The input time series data to fit the transform on.
            targets (torch.Tensor, optional): The target values associated with the time series data.
                        Has to be None if time_series is a Dataset.

        Returns
        -------
            None
        """
        if self.is_fitted:
            return
        if isinstance(time_series, Dataset):
            assert targets is None
            self._fit_dataset(time_series)
        else:
            self._fit(time_series, targets)
        self.is_fitted = True

    @overload
    def transform(self, time_series: torch.Tensor) -> torch.Tensor:
        ...

    @overload
    def transform(
        self, time_series: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def transform(self, time_series: Dataset) -> BaseDataset:
        ...

    def transform(
        self, time_series: Dataset | torch.Tensor, targets: torch.Tensor | None = None
    ) -> BaseDataset | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the transformation to the given time series data and optional targets.

        Args:
            time_series (Dataset | torch.Tensor): The input time series data to be transformed.
            targets (torch.Tensor, optional): The optional targets associated with the time series data.

        Returns
        -------
            BaseDataset: If a dataset is provided.
            torch.Tensor: If only the time series is provided.
            tuple[torch.Tensor, torch.Tensor]: If both the time series and targets are provided.

        Raises
        ------
            Exception: If the transform has not been fitted before it is used.
            RuntimeError: If transforming a dataset and targets are provided.

        """
        if self.is_fitted is False:
            raise Exception("Transform must be fitted before it can be used.")

        if isinstance(time_series, Dataset):
            if targets is not None:
                raise RuntimeError(
                    "If transforming a dataset, targets have to be None. "
                    "Do not provide targets to the transform method!"
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
        """
        Abstract method for a string representation of the transform.

        Returns
        -------
            str: The string representation of the object.
        """
        pass

    @abstractmethod
    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Abstract method for fitting the transform to the given time series data.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target values associated with the time series data.

        Returns
        -------
            None
        """
        pass

    @abstractmethod
    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Abstract method for performing a transformation on the input time series data.

        Args:
            time_series (torch.Tensor): The input time series to be transformed.
            targets (torch.Tensor, optional): The target values associated with the time series.
                Defaults to None.

        Returns
        -------
            tuple[torch.Tensor, None]: If only a time series is provided.
            tuple[torch.Tensor, torch.Tensor]: If both the time series and targets are provided.
        """
        pass

    @abstractmethod
    def _invert(self) -> None:
        """
        Abstract method to invert the transformation.

        This method should be implemented by subclasses to define how the transformation is inverted.

        Returns
        -------
            None
        """
        pass


class Compose(Transform):
    """
    A class representing a composition of multiple transforms.

    Args:
        transforms (list[Transform]): A list of transforms to be applied in sequence.

    Attributes
    ----------
        transforms (list[Transform]): The list of transforms in the composition.
    """

    def __init__(self, transforms: list[Transform]):
        all_fitted = all([t.is_fitted for t in transforms])
        super().__init__(is_fitted=all_fitted)
        self.transforms: list[Transform] = transforms

    def __add__(self, other: Transform) -> "Compose":
        """
        Add a transform to the composition.

        Args:
            other (Transform): The transform to be added.

        Returns
        -------
            Compose: A new composition with the added transform.
        """
        new_compose = Compose([*self.transforms, other])
        return new_compose

    def __getitem__(self, index: int) -> Transform:
        """
        Get a transform at the specified index.

        Args:
            index (int): The index of the transform to retrieve.

        Returns
        -------
            Transform: The transform at the specified index.
        """
        return self.transforms[index]

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fit the composition of transforms to the given time series and targets.

        The method fits and applies each transformation except the last one to the input data,
         to fitt all the transforms

        Args:
            time_series (torch.Tensor): The input time series.
            targets (torch.Tensor, optional): The target values. Defaults to None.

        Returns
        -------
            None
        """
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
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply the composition of transforms to the given time series and targets.

        Args:
            time_series (torch.Tensor): The input time series.
            targets (torch.Tensor, optional): The target values. Defaults to None.

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]: The transformed time series and targets.
        """
        if targets is None:
            for t in self.transforms:
                time_series = t.transform(time_series)
            return time_series, None
        else:
            for t in self.transforms:
                ts_transformed = t.transform(time_series, targets)
                time_series, targets = ts_transformed

            return time_series, targets

    def _invert(self) -> Transform:
        """
        Invert the composition of transforms.

        Returns
        -------
            Transform: The inverted composition of transforms.
        """
        return Compose([~t for t in self.transforms[::-1]])

    def __repr__(self) -> str:
        """
        Get a string representation of the composition.

        Returns
        -------
            str: The string representation of the composition.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
