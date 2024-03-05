"""
Spectral transforms for time series data.

Implemeted transforms:
    - FourierTransform: Applies the Fourier transform to time series data.
    - InverseFourierTransform: Applies the inverse Fourier transform to time series data.
"""

import torch

from .base_transforms import Transform


class FourierTransform(Transform):
    """Fourier Transform class that applies the Fourier transform to time series data."""

    def __init__(self) -> None:
        """Initialize a new instance of the FourierTransform class."""
        super().__init__(True)

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fits the Fourier transform to the given time series data.

        This method does not perform any fitting as the identity transformation
        does not require any parameters.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target values associated with the time series data.

        Returns
        -------
            None
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply the Fourier transform to the given time series data.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target values associated with the time series data.

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]: The Fourier transformed
            data and the targets (if provided).
        """
        ft = torch.fft.fft(time_series, norm="ortho")
        return ft, targets

    def _invert(self):
        """
        Return an instance of the InverseFourierTransform class.

        Returns
        -------
            InverseFourierTransform: An instance of the InverseFourierTransform class.
        """
        return InverseFourierTransform()

    def __repr__(self) -> str:
        """
        Return a string representation of the FourierTransform object.

        Returns
        -------
            str: A string representation of the FourierTransform object.
        """
        return "FourierTransform()"


class InverseFourierTransform(Transform):
    """Inverse Fourier transform."""

    def __init__(self):
        super().__init__(True)

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Fits the inverse Fourier transform to the given time series and targets.

        This method does not perform any fitting as the identity transformation does not
        require any parameters.

        Args:
            time_series (torch.Tensor): The input time series.
            targets (torch.Tensor, optional): The target values. Defaults to None.

        Returns
        -------
            None
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply the inverse Fourier transform to the given time series and targets.

        Args:
            time_series (torch.Tensor): The input time series.
            targets (torch.Tensor, optional): The target values. Defaults to None.

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor | None]: The transformed time series and targets.
        """
        ifft = torch.fft.ifft(time_series, norm="ortho")
        return ifft, targets

    def _invert(self):
        """
        Return the Fourier transform.

        Returns
        -------
            FourierTransform: The Fourier transform.
        """
        return FourierTransform()

    def __repr__(self) -> str:
        """
        Return a string representation of the InverseFourierTransform object.

        Returns
        -------
            str: The string representation of the InverseFourierTransform object.
        """
        return "InverseFourierTransform()"
