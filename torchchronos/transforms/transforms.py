import numpy as np
import torch
from .base import Transform, Compose


"""
This class transforms the labels of a dataset. It transforms arbitrary string/int labels to int labels form 0 to n_classes-1.
"""
class Identity(Transform):
    def __init__(self, is_fitted = True) -> None:
        super().__init__(is_fitted=is_fitted)

    def _fit(self, time_series: np.ndarray, y = None) -> None:
        pass

    def _transform(self, time_series: np.ndarray, y = None) -> np.ndarray:
        return time_series, y
    
    def _invert(self) -> Transform:
        return self

    def __repr__(self) -> str:
        return "Identity()"
    
class LabelTransform(Transform):
    def __init__(self, label_map = None) -> None:
        super().__init__(True if label_map is None else False)
        self.label_map = label_map

    def __repr__(self) -> str:
        return f"TransformLabels(label_map={self.label_map})"

    def _fit(self, time_series: np.ndarray, y) -> None:
        labels = np.unique(y)
        labels = np.sort(labels)
        self.label_map = {i: label for label, i in enumerate(labels)}

    def _transform(self, time_series: np.ndarray, y) -> tuple[np.ndarray, np.ndarray]:
        new_labels = np.array([self.label_map[label] for label in y])
        return time_series, new_labels
    
    def _invert(self) -> Transform:
        label_map = {v: k for k, v in self.label_map.items()}
        return LabelTransform(label_map)
    

class GlobalNormalize(Transform):
    def __init__(self) -> None:
        super().__init__()
        self.mean = None
        self.std = None

    def _fit(self, time_series: torch.Tensor, y = None) -> None:
        self.mean = torch.mean(time_series, axis = 0)
        self.std = torch.std(time_series, axis = 0)

    def _transform(self, time_series: torch.Tensor, y = None) -> tuple[np.ndarray, np.ndarray]:
        return (time_series - self.mean) / self.std, y

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"
    
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

    def _transform(self, time_series: torch.Tensor, y = None) -> tuple[np.ndarray, np.ndarray | None]:
        return time_series * self.scale, y
    
    def _invert(self) -> Transform:
        return Scale(1/self.scale)
    
    def __repr__(self) -> str:
        return f"Scale(scale={self.scale})"

class Shift(Transform):
    def __init__(self, shift: int) -> None:
        super().__init__(True)
        self.shift = shift

    def _fit(self, time_series, targets=None) -> None:
        pass

    def _transform(self, time_series: torch.Tensor, y = None) -> tuple[np.ndarray, np.ndarray]:
        return time_series + self.shift , y

    def __repr__(self) -> str:
        return f"Shift(shift={self.shift})"
    
    def _invert(self) -> Transform:
        return Shift(-self.shift)

