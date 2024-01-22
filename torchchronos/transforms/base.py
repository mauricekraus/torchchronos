from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import numpy as np
from aeon.transformations.base import BaseTransformer

# TODO: change to fit and _fit, transform and _transform for checking is fitted
# TODO: implement __invert__ for inverse transformations, add a attribute for the inverse transformation, for faster computation of double inverse and 
# TODO: implement scale and shift Transform, crop Transform, Dtype Transform, Reshape Transform

class Transform(ABC):
    def __init__(self, is_fitted: bool = False):
        self.is_fitted = is_fitted
        self._invert_transform = None

    def __add__(self, other):
        return Compose([self, other])
    
    def __invert__(self):
        if self._invert_transform is None:
            self._invert_transform = self._invert()
            self._invert_transform._invert_transform = self
        return self._invert_transform

    def save(self, name: str, path: Path | None = None):
        if path is None:
            path = Path(".cache/transforms/")
        
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / (name + ".pkl")

        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    def load(name: str, path: Path | None = None):
        if path is None:
            path = Path(".cache/transforms/")
        
        file_path = path / (name + ".pkl")

        with open(file_path, "rb") as file:
            transform = pickle.load(file)
        return transform

    def fit_transform(self, ts, targets=None):
        self.fit(ts, targets)
        ts, target = self.transform(ts, targets)
        return ts, target

    def fit(self, time_series, targets=None) -> None:
        if self.is_fitted:
            return
        self._fit(time_series, targets)
        self.is_fitted = True

    def transform(self, time_series, targets=None) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.is_fitted:
            raise Exception("Transform must be fitted before it can be used.")
        return self._transform(time_series, targets)

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

    def fit_transform(self, ts, targets=None):
        for t in self.transforms:
            ts, targets = t.fit_transform(ts, targets)

        return ts, targets

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
