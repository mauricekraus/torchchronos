from typing import Optional

import torch

from .base_transforms import Compose, Transform


class Identity(Transform):
    def __init__(self, is_fitted=True) -> None:
        super().__init__(is_fitted=is_fitted)

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return time_series, targets

    def _invert(self) -> Transform:
        return self

    def __repr__(self) -> str:
        return "Identity()"


class Normalize(Transform):
    def __init__(self, local:bool= False) -> None:
        super().__init__(local)
        self.local = local
        self.mean:Optional[torch.Tensor] = None
        self.std:Optional[torch.Tensor] = None

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        if self.local:
            return
        self.mean = torch.mean(time_series, 0, True)
        self.std = torch.std(time_series, 0, True) + 1e-5

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        if self.mean is None or self.std is None:
            # local normalization
            return f"{self.__class__.__name__}()"
        else:
            return f"{self.__class__.__name__}(mean={self.mean.shape}, std={self.std.shape})"

    def _invert(self) -> Transform:
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot invert transform before fitting.")

        return Compose([Scale(self.std), Shift(self.mean)], is_fitted=True)


class Scale(Transform):
    def __init__(self, scale: float | torch.Tensor) -> None:
        super().__init__(True)
        self.scale: float | torch.Tensor = scale

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return time_series * self.scale, targets

    def _invert(self) -> Transform:
        return Scale(1 / self.scale)

    def __repr__(self) -> str:
        if isinstance(self.scale, torch.Tensor):
            return f"Scale({self.scale.shape})"
        else:
            #single value
            return f"Scale({self.scale})"


class Shift(Transform):
    def __init__(self, shift: float | torch.Tensor) -> None:
        super().__init__(True)
        self.shift: float | torch.Tensor = shift

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return time_series + self.shift, targets

    def __repr__(self) -> str:
        if isinstance(self.shift, torch.Tensor):
            return f"Shift(shift={self.shift.shape})"
        else:
            #single value
            return f"Shift(shift={self.shift})"

    def _invert(self) -> Transform:
        return Shift(-self.shift)
