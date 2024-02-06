from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Any
import torch
import numpy as np
from aeon.transformations.base import BaseTransformer


# TODO: implement crop Transform, Dtype Transform, Reshape Transform

class Transform(ABC):
    def __init__(self, is_fitted: bool = False):
        self.is_fitted = is_fitted
        self._invert_transform = None

    def __call__(self, data:torch.tensor, y = None) -> Any:
        return self.transform(data, y)

    def __add__(self, other):
        return Compose([self, other])
    
    def __invert__(self):
        if self._invert_transform is None:
            self._invert_transform = self._invert()
            self._invert_transform._invert_transform = self
            self._invert_transform.is_fitted = True
        return self._invert_transform

    def save(self, name: str, path: Path | None = None):
        if not self.is_fitted:
            raise Exception("Transform must be fitted before it can be saved.")
        
        if path is None:
            path = Path(".cache/torchchronos/transforms")
        
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / (name + ".pkl")

        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    def load(name: str, path: Path | None = None):
        if path is None:
            path = Path(".cache/torchchronos/transforms")
        
        file_path = path / (name + ".pkl")

        with open(file_path, "rb") as file:
            transform = pickle.load(file)
        return transform

    def fit_transform(self, ts, targets=None):
        self.fit(ts, targets)
        ts, target = self.transform(ts, targets)
        self.is_fitted = True
        return ts, target

    def fit(self, time_series, targets=None) -> None:
        if self.is_fitted:
            return
        self._fit(time_series, targets)
        self.is_fitted = True

    def transform(self, time_series, targets=None, return_available_labels=False) -> tuple[np.ndarray, np.ndarray | None]:
        if self.is_fitted is False:
            raise Exception("Transform must be fitted before it can be used.")
        
        ts_transformed, targets_transformed = self._transform(time_series, targets)
        # if targets_transformed is None and return_available_labels is False:
        #     return ts_transformed

        return ts_transformed, targets_transformed

    @abstractmethod
    def _fit(self, time_series, targets=None) -> None:
        pass

    @abstractmethod
    def _transform(self, time_series, targets=None) -> tuple[np.ndarray, np.ndarray | None]:
        pass

    @abstractmethod
    def _invert(self):
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass



class Compose(Transform):
    def __init__(self, transforms: list[BaseTransformer | Transform], is_fitted=False):
        super().__init__(is_fitted=is_fitted)
        self.transforms = transforms
        for t in self.transforms:
            if not isinstance(t, Transform):
                assert hasattr(t, "fit") and callable(
                    t.fit
                ), "All transforms must have a fit method."
                assert hasattr(t, "transform") and callable(
                    t.transform
                ), "All transforms must have a transform method."

    def __add__(self, other):
        new_compose = Compose([*self.transforms, other])
        return new_compose

    def __getitem__(self, index):
        return self.transforms[index]

    def fit_transform(self, ts, targets=None):
        self.fit(ts, targets)
        ts, target = self.transform(ts, targets)
        self.is_fitted = True
        return ts, target

    def _fit(self, ts, targets=None) -> None:
        if self.transforms == []:
            return ts, targets
        
        for t in self.transforms[:-1]:
            ts, targets = t.fit_transform(ts, targets)

        self.transforms[-1].fit(ts, targets)
        self.is_fitted = True

    def _transform(self, ts, targets=None) -> tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            ts, targets = t.transform(ts, targets)
        return ts, targets
    
    def _invert(self) -> Transform:
        return Compose([~t for t in self.transforms[::-1]])

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
