import numpy as np

from .base import Transform


"""
This class transforms the labels of a dataset. It transforms arbitrary string/int labels to int labels form 0 to n_classes-1.
"""

class TransformLabels(Transform):
    def __init__(self) -> None:
        super().__init__()
        self.label_map = None

    def __repr__(self) -> str:
        return f"TransformLabels(label_map={self.label_map})"

    def fit(self, time_series: np.ndarray, y = None) -> Transform:
        labels = np.unique(y)
        labels = np.sort(labels)
        self.label_map = {i: label for label, i in enumerate(labels)}

    def transform(self, time_series: np.ndarray, y) -> np.ndarray:
        new_labels = np.array([self.label_map[label] for label in y])
        return time_series, new_labels