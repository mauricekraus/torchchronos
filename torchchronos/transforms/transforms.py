import numpy as np
import torch

from .base_transforms import Compose, Transform


class Identity(Transform):
    def __init__(self, is_fitted=True) -> None:
        super().__init__(is_fitted=is_fitted)

    def _fit(self, time_series: np.ndarray, y=None) -> None:
        pass

    def _transform(self, time_series: np.ndarray, y=None) -> np.ndarray:
        return time_series, y

    def _invert(self) -> Transform:
        return self

    def __repr__(self) -> str:
        return "Identity()"


class GlobalNormalize(Transform):
    def __init__(self) -> None:
        super().__init__()
        self.mean = None
        self.std = None

    def _fit(self, time_series: torch.Tensor, y=None) -> None:
        self.mean = torch.mean(time_series, 0, True)
        self.std = torch.std(time_series, 0, True) + 1e-5

    def _transform(
        self, time_series: torch.Tensor, y=None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        time_series = (time_series - self.mean) / self.std
        time_series[torch.isnan(time_series)] = 0
        return time_series, y

    def __repr__(self) -> str:
        return "Normalize()"
        # return f"Normalize(mean={self.mean.shape}, std={self.std.shape})"

    def _invert(self) -> Transform:
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot invert transform before fitting.")

        return Compose([Scale(self.std), Shift(self.mean)], is_fitted=True)


class Scale(Transform):
    def __init__(self, scale: float) -> None:
        super().__init__(True)
        self.scale: torch.Tensor | float = scale

    def _fit(self, time_series, targets=None) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, y=None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return time_series * self.scale, y

    def _invert(self) -> Transform:
        return Scale(1 / self.scale)

    def __repr__(self) -> str:
        if torch.is_tensor(self.scale):
            return f"Scale(scale={self.scale.shape})"
        return f"Scale(scale={self.scale})"


class Shift(Transform):
    def __init__(self, shift: int) -> None:
        super().__init__(True)
        self.shift = shift

    def _fit(self, time_series, targets=None) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, y=None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return time_series + self.shift, y

    def __repr__(self) -> str:
        if torch.is_tensor(self.shift):
            return f"Shift(shift={self.shift.shape})"
        return f"Shift(shift={self.shift})"

    def _invert(self) -> Transform:
        return Shift(-self.shift)
