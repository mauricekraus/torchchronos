"""Spectral transforms for time series data.

Implemeted transforms:
    - FourierTransform: Applies the Fourier transform to time series data.
    - InverseFourierTransform: Applies the inverse Fourier transform to time series data.
"""

import torch

from .base_transforms import Transform


class FourierTransform(Transform):
    """Fourier Transform class that applies the Fourier transform to time series data."""

    def __init__(self, norm="ortho") -> None:
        self.norm = norm
        super().__init__(True)

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Fits the Fourier transform to the given time series data.

        This method does not perform any fitting as the identity transformation
        does not require any parameters.

        Args:
            time_series: The input time series data.
            targets: The target values associated with the time series data.


        """

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the Fourier transform to the given time series data.

        Args:
            time_series: The input time series data.
            targets : The target values associated with the time series data.

        Returns:
            The Fourier transformed data and the targets (if provided).
        """
        ft = torch.fft.fft(time_series, norm=self.norm)
        return ft, targets

    def _invert(self):
        """Return an instance of the InverseFourierTransform class.

        Returns:
            An instance of the InverseFourierTransform class.
        """
        return InverseFourierTransform()

    def __repr__(self) -> str:
        return "FourierTransform()"


class InverseFourierTransform(Transform):
    """Inverse Fourier transform."""

    def __init__(self, norm="ortho"):
        self.norm = norm
        super().__init__(True)

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Fits the inverse Fourier transform to the given time series and targets.

        This method does not perform any fitting as the identity transformation does not
        require any parameters.

        Args:
            time_series: The input time series.
            targets: The target values.

        """

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the inverse Fourier transform to the given time series and targets.

        Args:
            time_series: The input time series.
            targets: The target values.

        Returns:
            The transformed time series and targets.
        """
        ifft = torch.fft.ifft(time_series, norm=self.norm)
        return ifft, targets

    def _invert(self):
        """Return the Fourier transform.

        Returns:
            The Fourier transform.
        """
        return FourierTransform()

    def __repr__(self) -> str:
        return "InverseFourierTransform()"
