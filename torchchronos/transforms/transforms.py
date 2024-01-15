import numpy as np

from .base import Transform


"""
This class transforms the labels of a dataset. It transforms arbitrary string/int labels to int labels form 0 to n_classes-1.
"""
class Identity(Transform):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, time_series: np.ndarray, y = None) -> Transform:
        pass

    def transform(self, time_series: np.ndarray, y = None) -> np.ndarray:
        return time_series, y

    def __repr__(self) -> str:
        return "Identity()"
    
class LabelTransform(Transform):
    def __init__(self) -> None:
        super().__init__()
        self.label_map = None

    def __repr__(self) -> str:
        return f"TransformLabels(label_map={self.label_map})"

    def fit(self, time_series: np.ndarray, y) -> None:
        labels = np.unique(y)
        labels = np.sort(labels)
        self.label_map = {i: label for label, i in enumerate(labels)}

    def transform(self, time_series: np.ndarray, y) -> tuple[np.ndarray, np.ndarray]:
        new_labels = np.array([self.label_map[label] for label in y])
        return time_series, new_labels
    
class LocalNormalize(Transform):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, time_series: np.ndarray, y = None) -> Transform:
        pass

    def transform(self, time_series: np.ndarray, y = None) -> np.ndarray:
        mean = np.mean(time_series, axis = 1)
        std = np.std(time_series, axis = 1)
        return (time_series - mean[:, None]) / std[:, None]

class GlobalNormalize(Transform):
    def __init__(self) -> None:
        super().__init__()
        self.mean = None
        self.std = None

    def fit(self, time_series: np.ndarray, y = None) -> Transform:
        self.mean = np.mean(time_series, axis = 0)
        self.std = np.std(time_series, axis = 0)

    def transform(self, time_series: np.ndarray, y = None) -> np.ndarray:
        return (time_series - self.mean) / self.std

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"

