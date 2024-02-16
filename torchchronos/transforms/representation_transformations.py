from typing import Optional

import numpy as np
import torch

from .base_transforms import Transform

"""
Representation transforms manipulate the representation of the data.
Examples are LabelTransform, and converting from complex to polar and vice versa.
"""


class LabelTransform(Transform):
    def __init__(self, label_map:Optional[dict[int, int]]=None) -> None:
        super().__init__(False if label_map is None else True)
        self.label_map = label_map

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        if targets is None:
            raise ValueError("Targets cannot be None.")
        
        targets.to(torch.int64)
        labels = np.unique(targets)
        labels = np.sort(labels)
        self.label_map = dict([(i, label) for label, i in enumerate(labels)])

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.label_map is None:
            raise ValueError("LabelTransform is not fitted. Please fit the transform first.")
        
        if targets is None:
            raise ValueError("Targets cannot be None.")
        
        new_targets = torch.tensor([self.label_map[int(label)] for label in targets], dtype=torch.int64)
        return time_series, new_targets

    def _invert(self) -> Transform:
        if self.label_map is None:
            raise ValueError("LabelTransform is not fitted. Please fit the transform first.")
        
        label_map = {value: key for key, value in self.label_map.items()}
        return LabelTransform(label_map)

    def __repr__(self) -> str:
        return f"TransformLabels(label_map={self.label_map})"

class ComplexToPolar(Transform):
    def __init__(self):
        super().__init__(True)

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        r = torch.abs(time_series)
        polar = torch.angle(time_series)

        ts_stacked = torch.stack((r, polar), dim=1).squeeze()
        return ts_stacked.type(torch.float32), targets

    def _invert(self):
        return PolarToComplex()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PolarToComplex(Transform):
    def __init__(self):
        super().__init__(True)

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        r = time_series[:, 0, :]
        polar = time_series[:, 1, :]

        ts_stacked = r * torch.exp(1j * polar)

        reshaped_ts = ts_stacked.reshape(time_series.shape[0], -1, time_series.shape[2])
        return reshaped_ts.type(torch.cfloat), targets

    def _invert(self):
        return ComplexToPolar()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CombineToComplex(Transform):
    def __init__(self):
        super().__init__(True)

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        samples_reshaped = time_series.reshape(
            time_series.shape[0], time_series.shape[1], -1, 2
        )
        complex_samples = (
            samples_reshaped[:, :, :, 0] + 1j * samples_reshaped[:, :, :, 1]
        )
        return complex_samples, targets

    def _invert(self):
        return SplitComplexToRealImag()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SplitComplexToRealImag(Transform):
    def __init__(self):
        super().__init__(True)

    def _fit(
            self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
            ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        flattened_time_series = torch.view_as_real(time_series)
        flattened_time_series = (
        flattened_time_series.type(torch.float32).flatten(1).unsqueeze(1))
        return flattened_time_series, targets


    def _invert(self):
        return CombineToComplex()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
