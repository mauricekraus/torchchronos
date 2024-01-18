from abc import ABC, abstractmethod
import numpy as np
from aeon.transformations.base import BaseTransformer

# TODO: change to fit and _fit, transform and _transform for checking is fitted
# TODO: implement __invert__ for inverse transformations, add a attribute for the inverse transformation, for faster computation of double inverse and 
# TODO: implement scale and shift Transform, crop Transform, Dtype Transform, Reshape Transform

class Transform(ABC):
    def __init__(self):
        self.is_fitted = False

    def __call__(self, ts, targets=None):
        self.transform(ts, targets)

    def __add__(self, other):
        return Compose([self, other])

    def fit_transform(self, ts, targets=None):
        self.fit(ts, targets)
        ts, target = self.transform(ts, targets)
        return ts, target

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def fit(self, time_series, targets=None) -> None:
        
        pass

    @abstractmethod
    def transform(self, time_series, targets=None):
        pass


class Compose(Transform):
    def __init__(self, transforms: list[BaseTransformer | Transform]):
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
        self.transforms.append(other)
        new_compose = Compose(self.transforms.append(other))
        return new_compose

    def fit_transform(self, ts, targets=None):
        for t in self.transforms:
            ts, targets = t.fit_transform(ts, targets)

        return ts, targets

    def fit(self, ts, targets=None) -> None: # TODO: check if empty -> raise Exception?
        for t in self.transforms[:-1]:
            ts, targets = t.fit_transform(ts, targets)

        self.transforms[-1].fit(ts, targets)
        self.is_fitted = True

    def transform(self, ts, targets=None) -> tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            ts, targets = t.transform(ts, targets)
        return ts, targets

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
