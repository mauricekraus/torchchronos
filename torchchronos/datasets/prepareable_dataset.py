"""Base Class for Datsets that have a prepare and a load step.

This class is the main building block for many further classes that inherit from it. The class has a perpare
and a load step. In the prepare step actions like downloading data, reading data, and preprocessing data
should be done. In the load step the data is loaded.

Note:
    When inheriting from this class keep the following in mind:
    - The inherited class can ignore the is_prepared and is_loaded attributes, all handeling is done in the
    base class.
    - In the load step the passed transformation has to be fitted. This can not be done here, since different
    classes will have different ways of saving the data such as numpy arrays, torch Tensors, or pandas
    DataFrames. Therefore, the fitting of the transformation has to be done in the inherited class.
-
"""

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset

from ..transforms import base_transforms, basic_transforms


class PrepareableDataset(ABC, Dataset):
    """A base class for prepareable datasets.

    Args:
        transform: The transform to be applied to the dataset.
        domain: The domain of the dataset.

    Attributes:
        is_prepared: Indicates whether the dataset has been prepared.
        is_loaded: Indicates whether the dataset has been loaded.
        _transform: The list of transforms to be applied to the dataset.
        domain: The domain of the dataset.

    """

    def __init__(
        self,
        transform: base_transforms.Transform = basic_transforms.Identity(),
        domain: str | None = None,
    ) -> None:
        self.is_prepared: bool = False
        self.is_loaded: bool = False
        self._transform: list[base_transforms.Transform] = transform
        self.domain: str | None = domain

    @property
    def transforms(self) -> base_transforms.Transform:
        """Get the transform to be applied to the dataset.

        Returns:
            Transform: The transform to be applied to the dataset.
        """
        return self._transform

    def __getitem__(self, idx: int) -> Any:
        """Get an item from the dataset.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            Any: The transformed time series and targets.

        Raises:
            NotPreparedError: If the dataset is not prepared before it is used.
            NotLoadedError: If the dataset is not loaded before it is used.
        """
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
        """Abstract method to get an item from the dataset.

        Args:
            idx : The index of the item to retrieve.

        Returns:
            Any: The retrieved item.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def prepare(self) -> None:
        """Prepare the dataset for usage."""
        if self.is_prepared:
            return
        self._prepare()
        self.is_prepared = True

    @abstractmethod
    def _prepare(self) -> None:
        """Abstract method to prepare the dataset.

        This method should be implemented by subclasses to perform any necessary data preparation steps.
        """
        pass

    def load(self) -> None:
        """Load the dataset for usage.

        Raises:
            NotPreparedError: If the dataset is not prepared before it is used.
            ValueError: If the transform is not fitted before the dataset is used.
        """
        if self.is_prepared is False:
            raise NotPreparedError("Dataset must be prepared before it can be loaded.")
        self._load()

        if self.transforms.is_fitted is False:
            raise ValueError(
                "The transform must be fitted before the dataset can be used."
                " This has to be done in the inherited class."
            )

        self.is_loaded = True

    @abstractmethod
    def _load(self) -> None:
        """Abstract method to load the dataset.

        This method should be implemented by subclasses to define how the dataset is loaded.
        """
        pass


class NotLoadedError(Exception):
    """Exception raised when a dataset is used before it is loaded."""

    pass


class NotPreparedError(Exception):
    """Exception raised when a dataset is used before it is prepared."""

    pass
