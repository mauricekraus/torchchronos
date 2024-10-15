"""Module for basic transformations of time series data.

The following transformations are implemented:
    - Identity: A transformation that returns the input time series and targets unchanged.
    - Normalize: Normalize the input time series data.
    - Scale: A transformation class that scales the input time series by a given factor.
    - Shift: A transformation class that shifts a time series data by a constant value or a tensor.
"""

import numpy as np
import torch

from torchchronos.transforms.base_transforms import Compose, Transform
from torchchronos.transforms.transformation_exceptions import NoInverseError


class Identity(Transform):
    """Identity tranform.

    Examples:
    The first example demonstrates how to use the Identity transform to return the input time series data
    unchanged.

    >>> import torch
    >>> time_series = torch.randn(10, 1, 100)  # 10 samples, 1 feature, 100 time points
    >>> identity_transform = Identity()
    >>> transformed_time_series = identity_transform(time_series)
    >>> transformed_time_series.shape
    torch.Size([10, 1, 100])

    The second example demonstrates how to use the Identity transform to return the input time series data and
    targets unchanged.

    >>> import torch
    >>> time_series = torch.randn(10, 1, 100)  # 10 samples, 1 feature, 100 time points
    >>> targets = torch.randn(10, 1)  # 10 samples, 1 target
    >>> identity_transform = Identity()
    >>> transformed_time_series, transformed_targets = identity_transform(time_series, targets)
    >>> transformed_time_series.shape
    torch.Size([10, 1, 100])
    >>> transformed_targets.shape
    torch.Size([10, 1])
    """

    def __init__(self) -> None:
        super().__init__(is_fitted=True)

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Fit the identity transformation.

        This method does not perform any fitting as the identity transformation
            does not require any parameters.

        Args:
            time_series: The input time series.
            targets: The input targets.
        """

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the identity transformation to the input time series and targets.

        Args:
            time_series: The input time series.
            targets: The input targets.

        Returns:
            The transformed time series and targets.
        """
        return time_series, targets

    def _invert(self) -> Transform:
        """Invert the identity transformation.

        Returns:
            The inverted transformation.
        """
        return self

    def __repr__(self) -> str:
        return "Identity()"


class Normalize(Transform):
    """Normalize the input time series data.

    Args:
        local: If True, perform local normalization. If False, perform global normalization.
            Defaults to False.

    Raises:
        RuntimeError: If attempting to transform or invert before fitting.

    Examples:
    The first example demonstrates how to use the Normalize transform to normalize the input over all time
    series data.

    >>> import torch
    >>> time_series = torch.randn(10, 1, 100)  # 10 samples, 1 feature, 100 time points
    >>> normalize_transform = Normalize()
    >>> normalize_transform.fit(time_series)
    >>> transformed_time_series = normalize_transform(time_series)

    The second example demonstrates how to use the Normalize transform to normalize the input over each time
    series data.

    >>> import torch
    >>> time_series = torch.arange(10).repeat(10, 1, 1).float()  # 10 samples, 1 feature, 100 time points
    >>> normalize_transform = Normalize(local=True)
    >>> transformed_time_series = normalize_transform(time_series)  # Does not require fitting
    >>> transformed_time_series[0, 0]
    tensor([-1.4863, -1.1560, -0.8257, -0.4954, -0.1651,  0.1651,  0.4954,  0.8257,
             1.1560,  1.4863])
    """

    def __init__(self, local: bool = False) -> None:
        super().__init__(local)
        self.local = local
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Fit the normalization parameters based on the input time series data.

        If self.local is True, nothing is done here
        If self.local is False, the mean and standard deviation are computed across the time dimension.

        Args:
            time_series: The input time series data.
            targets: The target values associated with the time series data.

        """
        if self.local:
            mean = torch.mean(time_series, 2, True).reshape(time_series.shape[0], time_series.shape[1], 1)
            std = torch.std(time_series, 2, True).reshape(time_series.shape[0], time_series.shape[1], 1) + 1e-5
            self.std = torch.nan_to_num(self.std, nan=1e-5)
        else:
            self.mean = torch.from_numpy(np.nanmean(time_series, axis=0, keepdims=True))
            self.std = torch.from_numpy(np.nanstd(time_series, axis=0, keepdims=True, ddof=1) + 1e-5)
            self.std = torch.nan_to_num(self.std, nan=1e-5)

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the normalization transformation to the input time series data.

        Args:
            time_series: The input time series data.
            targets : The target values associated with the time series data.


        Returns:
            The normalized time series data and the targets (if provided).

        """
        if self.mean is None or self.std is None:
                raise RuntimeError("Cannot transform before fitting.")
            
        if self.local:
            time_series = (time_series - mean) / std
            time_series[torch.isnan(time_series)] = 0
            return time_series, targets
        else:
            time_series = (time_series - self.mean) / self.std
            return time_series, targets

    def __repr__(self) -> str:
        mode = "local" if self.local else "global"
        if self.mean is None or self.std is None:
            return f"{self.__class__.__name__}({mode})"
        else:
            return f"{self.__class__.__name__}({mode}, mean={self.mean.shape}, std={self.std.shape})"

    def _invert(self) -> Transform:
        """Invert the normalization transform.

        Returns:
            The inverted normalization transform.

        Raises:
            RuntimeError: If attempting to invert before fitting.

        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot invert transform before fitting.")

        return Compose([Scale(self.std), Shift(self.mean)])


class Scale(Transform):
    """A transformation class that scales the input time series by a given factor.

    Args:
        scale: The scaling factor to apply to the time series.

    Examples:
    The first example demonstrates how to use the Scale transform to scale the input time series data.

    >>> import torch
    >>> time_series = torch.arange(10).repeat(10, 1, 1).float()  # 10 samples, 1 feature, 10 time points
    >>> scale_transform = Scale(2)
    >>> transformed_time_series = scale_transform(time_series)  # Does not require fitting
    >>> transformed_time_series[0, 0]
    tensor([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.])

    The second example demonstrates how to use the Scale transform to scale each time step individually.

    >>> import torch
    >>> time_series = torch.arange(10).repeat(10, 1, 1).float()  # 10 samples, 1 feature, 10 time points
    >>> scale_transform = Scale(torch.arange(1, 11).float())
    >>> transformed_time_series = scale_transform(time_series)  # Does not require fitting
    >>> transformed_time_series[0, 0]
    tensor([ 0.,  2.,  6., 12., 20., 30., 42., 56., 72., 90.])
    """

    def __init__(self, scale: float | torch.Tensor) -> None:
        super().__init__(True)
        self.scale: float | torch.Tensor = scale

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Fit the scaling transformation to the input time series.

        This method does not perform any fitting as the identity transformation does not
        require any parameters.

        Args:
            time_series: The input time series to fit the scaling transformation to.
            targets: The target values associated with the time series.

        """

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the scaling transformation to the input time series.

        Args:
            time_series: The input time series to apply the scaling transformation to.
            targets: The target values associated with the time series.


        Returns:
            The scaled time series and the targets (if provided).
        """
        return time_series * self.scale, targets

    def _invert(self) -> Transform:
        """Return the inverse transformation of the scaling transformation.

        Returns:
            The inverse scaling transformation.
        """
        return Scale(1 / self.scale)

    def __repr__(self) -> str:
        """Return a string representation of the Scale transform.

        If the scale is a tensor, the shape of the tensor is included in the string representation.
        If the scale is a single value, only the value is included in the string representation.

        Returns:
            A string representation of the Scale transform.
        """
        if isinstance(self.scale, torch.Tensor):
            return f"Scale({self.scale.shape})"
        else:
            # single value
            return f"Scale({self.scale})"


class Shift(Transform):
    """A transformation class that shifts a time series data by a constant value or a tensor.

    Args:
        shift: The amount by which the time series data is shifted.

    Examples:
        The first example demonstrates how to use the Shift transform to shift the input time series data.

        >>> import torch
        >>> time_series = torch.arange(10).repeat(10, 1, 1).float()  # 10 samples, 1 feature, 10 time points
        >>> shift_transform = Shift(2)
        >>> transformed_time_series = shift_transform(time_series)  # Does not require fitting
        >>> transformed_time_series[0, 0]
        tensor([ 2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])

        The second example demonstrates how to use the Shift transform to Shift each time step individually.

        >>> import torch
        >>> time_series = torch.arange(10).repeat(10, 1, 1).float()  # 10 samples, 1 feature, 10 time points
        >>> shift_transform = Shift(torch.arange(1, 11).float())
        >>> transformed_time_series = shift_transform(time_series)  # Does not require fitting
        >>> transformed_time_series[0, 0]
        tensor([ 1.,  3.,  5.,  7.,  9., 11., 13., 15., 17., 19.])
    """

    def __init__(self, shift: float | torch.Tensor) -> None:
        super().__init__(True)
        self.shift: float | torch.Tensor = shift

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Fits the shift transformation.

        This method does not perform any fitting as the identity transformation does not
        require any parameters.

        Args:
            time_series: The input time series data.
            targets: The target data (if applicable).
        """

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transform the time series data by adding the shift value.

        Args:
            time_series: The input time series data.
            targets: The target data (if applicable).

        Returns:
            The transformed time series data and the targets (if provided).
        """
        return time_series + self.shift, targets

    def __repr__(self) -> str:
        if isinstance(self.shift, torch.Tensor):
            return f"Shift(shift={self.shift.shape})"
        else:
            # single value
            return f"Shift(shift={self.shift})"

    def _invert(self) -> "Shift":
        """Return the inverse transformation of the shift operation.

        Returns:
            The inverse transformation of the shift operation.
        """
        return Shift(-self.shift)


class NaNToNumber(Transform):
    """A transformation to replace all NaNs with a fixed number.

    Args:
        replacemnt: The number that NaN is replaced with.

    Examples:
        The first example demonstrates how to use the Shift transform to shift the input time series data.

        >>> import torch
        >>> data = [0, 1, 3, float("NaN"), float("NaN")]
        >>> time_series = torch.tensor(data).float()
        >>> replace_transform = NaNToNumber(9)
        >>> transformed_time_series = replace_transform(time_series)  # Does not require fitting
        >>> transformed_time_series
        tensor([0., 1., 3., 9., 9.])

    """

    def __init__(self, replacement: float = 0):
        self.replacement = replacement

        super().__init__(True)

    def _fit(self, time_series, targets):
        pass

    def _transform(self, time_series, targets):
        time_series[torch.isnan(time_series)] = self.replacement
        return time_series, targets

    def _invert(self):
        error_message = (
            "This transform can not be reversed. To reverse this, the transform would have to "
            "safe the positions of the replaced numbers. This is not done."
        )
        raise NoInverseError(error_message)

    def __repr__(self):
        return f"NaNToNumber(replacement={self.replacement})"
