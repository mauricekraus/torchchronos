from typing import Optional

import torch
import numpy as np

from .base_transforms import Transform
from .transformation_exceptions import NoInverseError


class ToTorchTensor(Transform):
    """
    A transformation class to convert data into torch tensors.

    This transformation is applied to both time series data and optional target data.
    """

    def __init__(self):
        """
        Initializes the ToTorchTensor transformation.
        """
        super().__init__(True)

    def _fit(self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        """
        Fits the transformation.

        This method does not perform any fitting as the identity transformation does not require any parameters.

        Args:
            time_series (torch.Tensor): The time series data.
            targets (torch.Tensor, optional): The target data. Defaults to None.

        Returns:
            None
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Converts the input time series and target data (if provided) into torch tensors.

        Args:
            time_series (torch.Tensor): The time series data.
            targets (torch.Tensor, optional): The target data. Defaults to None.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: The transformed time series and target data (if provided).
        """
        if (torch.is_tensor(time_series) and targets is None) or (
            torch.is_tensor(time_series) and torch.is_tensor(targets)
        ):
            return time_series, targets

        if targets is None:
            targets = None
        elif isinstance(targets[0], str):
            np_array = np.array(targets).astype(np.float32)
            targets = torch.from_numpy(np_array).float()
        else:
            targets = torch.tensor(targets).float()

        return torch.tensor(time_series).float(), targets

    def _invert(self) -> Transform:
        """
        Raises an exception since inversion is not supported for this transformation.

        Raises:
            NoInverseError: The inversion of the transformation is not supported.
        """
        raise NoInverseError()

    def __repr__(self) -> str:
        """
        Returns a string representation of the transformation.

        Returns:
            str: The string representation of the transformation.
        """
        return f"{self.__class__.__name__}()"


class ToNumpyArray(Transform):
    """
    A transformation class to convert data into numpy arrays.

    This transformation is applied to both time series data and optional target data.
    """

    def __init__(self):
        """
        Initializes the ToNumpyArray transformation.
        """
        super().__init__(True)

    def _fit(self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        """
        Fits the transformation.

        This method does not perform any fitting as the identity transformation does not require any parameters.

        Args:
            time_series (torch.Tensor): The time series data.
            targets (torch.Tensor, optional): The target data. Defaults to None.

        Returns:
            None
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Converts the input time series and target data (if provided) into numpy arrays.

        Args:
            time_series (torch.Tensor): The time series data.
            targets (torch.Tensor, optional): The target data. Defaults to None.

        Returns:
            tuple[np.ndarray, Optional[np.ndarray]]: The transformed time series and target data (if provided).
        """
        if targets is None:
            return time_series.numpy(), None
        else:
            return time_series.numpy(), targets.numpy()

    def _invert(self):
        """
        Raises an exception since inversion is not supported for this transformation.

        Raises:
            NoInverseError: The inversion of the transformation is not supported.
        """
        raise NoInverseError()

    def __repr__(self) -> str:
        """
        Returns a string representation of the transformation.

        Returns:
            str: The string representation of the transformation.
        """
        return f"{self.__class__.__name__}()"


class To(Transform):
    """
    A transformation class to convert data into a specific torch data type.

    This transformation is applied to both time series data and optional target data.

    Attributes:
        torch_attribute (torch.dtype): The torch data type to convert the data to.
    """

    def __init__(self, torch_attribute):
        """
        Initializes the To transformation.

        Args:
            torch_attribute (torch.dtype): The torch data type to convert the data to.
        """
        super().__init__(True)
        self.torch_attribute = torch_attribute

    def _fit(self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        """
        Fits the transformation.

        This method does not perform any fitting as the identity transformation does not require any parameters.

        Args:
            time_series (torch.Tensor): The time series data.
            targets (torch.Tensor, optional): The target data. Defaults to None.

        Returns:
            None
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Converts the input time series and target data (if provided) into the specified torch data type.

        Args:
            time_series (torch.Tensor): The time series data.
            targets (torch.Tensor, optional): The target data. Defaults to None.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: The transformed time series and target data (if provided).
        """
        if targets is None:
            return time_series.to(self.torch_attribute), None
        else:
            return time_series.to(self.torch_attribute), targets.to(self.torch_attribute)

    def _invert(self):
        """
        Raises an exception since inversion is not supported for this transformation.

        Raises:
            NoInverseError: The inversion of the transformation is not supported.
        """
        raise NoInverseError()

    def __repr__(self) -> str:
        """
        Returns a string representation of the transformation.

        Returns:
            str: The string representation of the transformation.
        """
        return f"{self.__class__.__name__}(attribute={self.torch_attribute})"
