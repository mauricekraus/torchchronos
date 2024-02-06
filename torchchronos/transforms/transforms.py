import numpy as np
import torch
from .base import Transform, Compose


"""
This class transforms the labels of a dataset. It transforms arbitrary string/int labels to int labels form 0 to n_classes-1.
"""

class Crop(Transform):
    def __init__(self, start, end) -> None:
        super().__init__(True)
        self.start = start
        self.end = end

    def _fit(self, time_series, y=None) -> None:
        pass

    def _transform(self, time_series: np.ndarray, y=None) -> tuple[np.ndarray, np.ndarray]:
        return time_series[:, :, self.start:self.end], y

    def _invert(self) -> Transform:
        return self

    def __repr__(self) -> str:
        return f"Crop(start={self.start}, end={self.end})"

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


class LabelTransform(Transform):
    def __init__(self, label_map=None) -> None:
        super().__init__(False if label_map is None else True)
        self.label_map = label_map

    def __repr__(self) -> str:
        if self.is_fitted:
            return f"LabelTransform()"
        return f"TransformLabels(label_map={self.label_map})"

    def _fit(self, time_series: np.ndarray, y) -> None:
        labels = np.unique(y)
        labels = np.sort(labels)
        self.label_map = {i: label for label, i in enumerate(labels)}
        print(self.label_map)

    def _transform(self, time_series: np.ndarray, y) -> tuple[np.ndarray, np.ndarray]:
        new_labels = np.array([self.label_map[label] for label in y])
        return time_series, new_labels

    def _invert(self) -> Transform:
        label_map = {value: key for key, value in self.label_map.items()}
        return LabelTransform(label_map)


class GlobalNormalize(Transform):
    def __init__(self) -> None:
        super().__init__()
        self.mean = None
        self.std = None

    def _fit(self, time_series: torch.Tensor, y=None) -> None:
        self.mean = torch.mean(time_series, 0, True)
        self.std = torch.std(time_series, 0, True) + 1e-5

    def _transform(
        self, time_series: torch.Tensor, y=None) -> tuple[np.ndarray, np.ndarray | None]:
        time_series = (time_series - self.mean) / self.std
        time_series[torch.isnan(time_series)] = 0
        return time_series, y

    def __repr__(self) -> str:
        return "Normalize()"
        #return f"Normalize(mean={self.mean.shape}, std={self.std.shape})"

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
    ) -> tuple[np.ndarray, np.ndarray]:
        return time_series + self.shift, y

    def __repr__(self) -> str:
        if torch.is_tensor(self.shift):
            return f"Shift(shift={self.shift.shape})"
        return f"Shift(shift={self.shift})"

    def _invert(self) -> Transform:
        return Shift(-self.shift)
    
    
    

class ToTorchTensor(Transform):
    def __init__(self):
        super().__init__()

    def _fit(self, time_series, targets=None) -> None:
        pass

    def _transform(self, time_series: torch.Tensor, targets=None) -> torch.Tensor:
        targets = targets if targets is None else torch.tensor(targets)
        if torch.is_tensor(time_series):
            return time_series, targets
        return torch.tensor(time_series).float(), targets

    def _invert(self):
        return self # TODO: maybe raise error
    
    def __repr__(self) -> str:
        return "ToTorchTensor()"
