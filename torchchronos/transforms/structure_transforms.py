import numpy as np
import torch

from .base_transforms import Transform
from .transformation_exceptions import NoInverseError

"""
Structure transforms manipulate the structure of the underlying data.
Examples are cropping, and Padding.
"""


class Crop(Transform):
    def __init__(self, start, end) -> None:
        super().__init__()
        self.start = start
        self.end = end

    def _fit(self, time_series, y=None) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("Start and end must be positive integers")
        if self.start >= self.end:
            raise ValueError("Start must be less than end")
        if self.end > time_series.shape[-1]:
            raise ValueError("End must be less than the length of the time series")

    def _transform(
        self, time_series: np.ndarray, y=None
    ) -> tuple[np.ndarray, np.ndarray]:
        return time_series[:, :, self.start : self.end], y

    def _invert(self) -> Transform:
        raise NoInverseError("Crop transformation is not invertible")

    def __repr__(self) -> str:
        return f"Crop(start={self.start}, end={self.end})"


class PadFront(Transform):
    def __init__(self, length):
        super().__init__()
        self.length = length
        self.time_series_length: int | None = None

    def _fit(self, time_series, targets=None) -> None:
        self.time_series_length = time_series.shape[-1]

    def _transform(self, ts, y=None):
        zeros = torch.zeros((ts.shape[0], ts.shape[1], self.length))
        return torch.cat([zeros, ts], dim=2), y

    def _invert(self):
        return Crop(self.length, self.time_series_length + self.length)

    def __repr__(self):
        return self.__class__.__name__ + f"(length={self.length})"


class PadBack(Transform):
    def __init__(self, length):
        super().__init__()
        self.length = length
        self.time_series_length: int | None = None

    def _fit(self, time_series, targets=None) -> None:
        self.time_series_length = time_series.shape[-1]

    def _transform(self, ts, y=None):
        zeros = torch.zeros((ts.shape[0], ts.shape[1], self.length))
        return torch.cat([ts, zeros], dim=2), y

    def _invert(self):
        return Crop(0, self.time_series_length)

    def __repr__(self):
        return self.__class__.__name__ + f"(length={self.length})"


class Filter(Transform):
    def __init__(self, filter: callable):
        super().__init__(True)
        self.filter: callable = filter

    def _fit(self, time_series, targets=None) -> None:
        pass

    def _transform(self, time_series: torch.Tensor, targets=None) -> torch.Tensor:
        indecies = []
        if targets is None:
            for i in range(len(time_series)):
                if self.filter(time_series[i], None):
                    indecies.append(i)
            return time_series[indecies], None
        else:
            for i in range(len(time_series)):
                if self.filter(time_series[i], targets[i]):
                    indecies.append(i)
            return time_series[indecies], targets[indecies]

    def _invert(self):
        raise NoInverseError("Filter transformation is not invertible")

    def __repr__(self) -> str:
        return f"{__class__.__name__}()"
