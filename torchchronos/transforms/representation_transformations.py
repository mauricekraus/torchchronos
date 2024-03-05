"""Module for transformations to convert the data into different representations."""

import numpy as np
import torch

from .base_transforms import Transform


class LabelTransform(Transform):
    """
    Class to map labels to integer indices.

    Attributes
    ----------
        label_map (dict[int, int], optional): A mapping of labels to integer indices.
    """

    def __init__(self, label_map: dict[int, int] | None = None) -> None:
        """
        Initialize the LabelTransform.

        Args:
            label_map (dict[int, int], optional): A mapping of labels to integer indices. Defaults to None.
        """
        super().__init__(False if label_map is None else True)
        self.label_map = label_map

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fit the LabelTransform.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target data.

        Returns
        -------
            None

        Raises
        ------
            ValueError: If targets is None.
        """
        if targets is None:
            raise ValueError("Targets cannot be None.")

        targets.to(torch.int64)
        labels = np.unique(targets)
        labels = np.sort(labels)
        self.label_map = dict([(i, label) for label, i in enumerate(labels)])

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Transform the data by mapping labels to integer indices.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target data.

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]: The transformed time series data and
            the transformed targets.

        Raises
        ------
            ValueError: If LabelTransform is not fitted.
        """
        if self.label_map is None:
            raise ValueError("LabelTransform is not fitted. Please fit the transform first.")

        if targets is None:
            raise ValueError("Targets cannot be None.")

        new_targets = torch.tensor([self.label_map[int(label)] for label in targets], dtype=torch.int64)
        return time_series, new_targets

    def _invert(self) -> Transform:
        """
        Invert the LabelTransform.

        Returns
        -------
            Transform: The inverted LabelTransform.

        Raises
        ------
            ValueError: If LabelTransform is not fitted.
        """
        if self.label_map is None:
            raise ValueError("LabelTransform is not fitted. Please fit the transform first.")

        label_map = {value: key for key, value in self.label_map.items()}
        return LabelTransform(label_map)

    def __repr__(self) -> str:
        """
        Return a string representation of the LabelTransform.

        Returns
        -------
            str: String representation of the LabelTransform.
        """
        return f"TransformLabels(label_map={self.label_map})"


class ComplexToPolar(Transform):
    """Class to convert complex numbers to polar representation."""

    def __init__(self):
        """Initialize the ComplexToPolar transformation."""
        super().__init__(True)

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fit the ComplexToPolar transformation.

        This method does not perform any fitting as the identity transformation does not
        require any parameters.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target data.
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Convert the input complex numbers to polar representation.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target data.

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]: The transformed time series data and the targets.
        """
        r = torch.abs(time_series)
        polar = torch.angle(time_series)

        ts_stacked = torch.stack((r, polar), dim=1).squeeze()
        return ts_stacked.type(torch.float32), targets

    def _invert(self):
        """
        Invert the ComplexToPolar transformation.

        Returns
        -------
            PolarToComplex: The inverted transformation.
        """
        return PolarToComplex()

    def __repr__(self) -> str:
        """
        Return a string representation of the transformation.

        Returns
        -------
            str: String representation of the ComplexToPolar transformation.
        """
        return f"{self.__class__.__name__}()"


class PolarToComplex(Transform):
    """Class to convert polar representation to complex numbers."""

    def __init__(self):
        """Initialize the PolarToComplex transformation."""
        super().__init__(True)

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fit the PolarToComplex transformation.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target data.

        Returns
        -------
            None
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Transform the data from polar representation to complex numbers.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target data.

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]: The transformed time series data and the targets.
        """
        r = time_series[:, 0, :]
        polar = time_series[:, 1, :]

        ts_stacked = r * torch.exp(1j * polar)

        reshaped_ts = ts_stacked.reshape(time_series.shape[0], -1, time_series.shape[2])
        return reshaped_ts.type(torch.cfloat), targets

    def _invert(self):
        """
        Invert the PolarToComplex transformation.

        Returns
        -------
            ComplexToPolar: The inverted transformation.
        """
        return ComplexToPolar()

    def __repr__(self) -> str:
        """
        Return a string representation of the PolarToComplex transformation.

        Returns
        -------
            str: String representation of the PolarToComplex transformation.
        """
        return f"{self.__class__.__name__}()"


class CombineToComplex(Transform):
    """Class to combine real and imaginary parts into complex numbers."""

    def __init__(self):
        """Initialize the CombineToComplex transformation."""
        super().__init__(True)

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fit the CombineToComplex transformation.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target to the data.
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Transform the data by combining real and imaginary parts into complex numbers.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor | None): The target to the data.

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]: The transformed time series data and the targets.
        """
        samples_reshaped = time_series.reshape(time_series.shape[0], time_series.shape[1], -1, 2)
        complex_samples = samples_reshaped[:, :, :, 0] + 1j * samples_reshaped[:, :, :, 1]
        return complex_samples, targets

    def _invert(self):
        """
        Invert the CombineToComplex transformation.

        Returns
        -------
            SplitComplexToRealImag: The inverted transformation.
        """
        return SplitComplexToRealImag()

    def __repr__(self) -> str:
        """
        Return a string representation of the CombineToComplex transformation.

        Returns
        -------
            str: String representation of the CombineToComplex transformation.
        """
        return f"{self.__class__.__name__}()"


class SplitComplexToRealImag(Transform):
    """Class to split complex numbers into real and imaginary parts."""

    def __init__(self):
        """Initialize the SplitComplexToRealImag transformation."""
        super().__init__(True)

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fit the SplitComplexToRealImag transformation.

        This method does not perform any fitting as the identity transformation does not
        require any parameters.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target data.

        Returns
        -------
            None
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Split a complex time series into real and imaginary parts.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target data.

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]: The transformed time series data and the targets.
        """
        flattened_time_series = torch.view_as_real(time_series)
        flattened_time_series = flattened_time_series.type(torch.float32).flatten(1).unsqueeze(1)
        return flattened_time_series, targets

    def _invert(self):
        """
        Invert the SplitComplexToRealImag transformation.

        Returns
        -------
            CombineToComplex: The inverted transformation.
        """
        return CombineToComplex()

    def __repr__(self) -> str:
        """
        Return a string representation of the transformation.

        Returns
        -------
            str: String representation of the SplitComplexToRealImag transformation.
        """
        return f"{self.__class__.__name__}()"
