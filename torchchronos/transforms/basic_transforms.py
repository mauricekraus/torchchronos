"""Module for basic transformations of time series data.

The following transformations are implemented:
    - Identity: A transformation that returns the input time series and targets unchanged.
    - Normalize: Normalize the input time series data.
    - Scale: A transformation class that scales the input time series by a given factor.
- Shift: A transformation class that shifts a time series data by a constant value or a tensor.
"""

import torch

from . import base_transforms


class Identity(base_transforms.Transform):
    """Identity tranform."""

    def __init__(self) -> None:
        super().__init__(is_fitted=True)

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fit the identity transformation.

        This method does not perform any fitting as the identity transformation
            does not require any parameters.

        Args:
            time_series (torch.Tensor): The input time series.
            targets (torch.Tensor, optional): The input targets. Defaults to None.
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply the identity transformation to the input time series and targets.

        Args:
            time_series (torch.Tensor): The input time series.
            targets (torch.Tensor, optional): The input targets. Defaults to None.

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]: The transformed time series and targets.
        """
        return time_series, targets

    def _invert(self) -> base_transforms.Transform:
        """
        Invert the identity transformation.

        Returns
        -------
            Transform: The inverted transformation.
        """
        return self

    def __repr__(self) -> str:
        """Return a string representation of the Identity object."""
        return "Identity()"


class Normalize(base_transforms.Transform):
    """
    Normalize the input time series data.

    Args:
        local (bool, optional): If True, perform local normalization. If False, perform global normalization.
            Defaults to False.

    Attributes
    ----------
        local (bool): Indicates whether local or global normalization is performed.
        mean (torch.Tensor, optional): The mean values used for normalization. None if not yet fitted.
        std (torch.Tensor, optional): The standard deviation values used for normalization.
                None if not yet fitted.

    Raises
    ------
        RuntimeError: If attempting to transform or invert before fitting.

    """

    def __init__(self, local: bool = False) -> None:
        super().__init__(local)
        self.local = local
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fit the normalization parameters based on the input time series data.

        If self.local is True, nothing is done here
        If self.local is False, the mean and standard deviation are computed across the time dimension.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target values associated with the time series data.
                Defaults to None.

        Returns
        -------
            None

        """
        if self.local:
            return
        self.mean = torch.mean(time_series, 0, True)
        self.std = torch.std(time_series, 0, True) + 1e-5

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply the normalization transformation to the input time series data.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target values associated with the time series data.
                Defaults to None.

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]:
                    The normalized time series data and the targets (if provided).

        """
        if self.local:
            mean = torch.mean(time_series, 2, True)
            std = torch.std(time_series, 2, True) + 1e-5
            time_series = (time_series - mean) / std
            time_series[torch.isnan(time_series)] = 0
            return time_series, targets
        else:
            if self.mean is None or self.std is None:
                raise RuntimeError("Cannot transform before fitting.")

            time_series = (time_series - self.mean) / self.std
            time_series[torch.isnan(time_series)] = 0
            return time_series, targets

    def __repr__(self) -> str:
        """
        Return a string representation of the Normalize object.

        Returns
        -------
            str: The string representation of the Normalize object.

        """
        if self.mean is None or self.std is None:
            mode = "local" if self.local else "global"
            return f"{self.__class__.__name__}({mode})"
        else:
            return f"{self.__class__.__name__}({mode}, mean={self.mean.shape}, std={self.std.shape})"

    def _invert(self) -> base_transforms.Transform:
        """
        Invert the normalization transform.

        Returns
        -------
            Transform: The inverted normalization transform.

        Raises
        ------
            RuntimeError: If attempting to invert before fitting.

        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot invert transform before fitting.")

        return base_transforms.Compose([Scale(self.std), Shift(self.mean)])


class Scale(base_transforms.Transform):
    """
    A transformation class that scales the input time series by a given factor.

    Args:
        scale (float or torch.Tensor): The scaling factor to apply to the time series.

    Attributes
    ----------
        scale (float or torch.Tensor): The scaling factor to apply to the time series.

    """

    def __init__(self, scale: float | torch.Tensor) -> None:
        """
        Initialize the Scale transformation.

        Args:
            scale (float or torch.Tensor): The scaling factor to apply to the time series.
        """
        super().__init__(True)
        self.scale: float | torch.Tensor = scale

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fit the scaling transformation to the input time series.

        This method does not perform any fitting as the identity transformation does not
        require any parameters.

        Args:
            time_series (torch.Tensor): The input time series to fit the scaling transformation to.
            targets (torch.Tensor, optional): The target values associated with the time series.
                Defaults to None.
        """
        pass

    def _transform(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply the scaling transformation to the input time series.

        Args:
            time_series (torch.Tensor): The input time series to apply the scaling transformation to.
            targets (torch.Tensor, optional): The target values associated with the time series.
                Defaults to None.

        Returns
        -------
            torch.Tensor: The scaled time series.
        """
        pass

    def _invert(self) -> "Scale":
        """
        Return the inverse transformation of the scaling transformation.

        Returns
        -------
            Scale: The inverse scaling transformation.
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply the scaling transformation to the input time series.

        Args:
            time_series (torch.Tensor): The input time series to apply the scaling transformation to.
            targets (torch.Tensor, optional): The target values associated with the time series.
                Defaults to None.

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]: The scaled time series and the targets (if provided).
        """
        pass
        return time_series * self.scale, targets

    def _invert(self) -> base_transforms.Transform:
        """
        Return the inverse transformation of the scaling transformation.

        Returns
        -------
            Transform: The inverse scaling transformation.
        """
        return Scale(1 / self.scale)

    def __repr__(self) -> str:
        """
        Return a string representation of the Scale transform.

        If the scale is a tensor, the shape of the tensor is included in the string representation.
        If the scale is a single value, only the value is included in the string representation.

        Returns
        -------
            str: A string representation of the Scale transform.
        """
        if isinstance(self.scale, torch.Tensor):
            return f"Scale({self.scale.shape})"
        else:
            # single value
            return f"Scale({self.scale})"


class Shift(base_transforms.Transform):
    """
    A transformation class that shifts a time series data by a constant value or a tensor.

    Args:
        shift (float or torch.Tensor): The amount by which the time series data is shifted.

    Attributes
    ----------
        shift (float or torch.Tensor): The shift value.
    """

    def __init__(self, shift: float | torch.Tensor) -> None:
        """
        Initialize the Shift transformation.

        Args:
            shift (float or torch.Tensor): The amount by which the time series data is shifted.
        """
        super().__init__(True)
        self.shift: float | torch.Tensor = shift

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fits the shift transformation.

        This method does not perform any fitting as the identity transformation does not
        require any parameters.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor | None): The target data (if applicable).
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Transform the time series data by adding the shift value.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor | None): The target data (if applicable).

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]:
                The transformed time series data and the targets (if provided).
        """
        return time_series + self.shift, targets

    def __repr__(self) -> str:
        """
        Return a string representation of the shift transformation.

        Returns
        -------
            str: String representation of the Shift transformation.
        """
        if isinstance(self.shift, torch.Tensor):
            return f"Shift(shift={self.shift.shape})"
        else:
            # single value
            return f"Shift(shift={self.shift})"

    def _invert(self) -> "Shift":
        """
        Return the inverse transformation of the shift operation.

        Returns
        -------
            Shift: The inverse transformation of the shift operation.
        """
        return Shift(-self.shift)
