from abc import ABC, abstractmethod
import torch
from aeon.transformations.base import BaseTransformer


class Transform(ABC):

    def __init__(self):
        self.is_fitted = False

    def __call__(self, ts, targets = None):
        self.transform(ts, targets)

    def __add__(self, other):
        return Compose([self, other])
    
    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def fit(self, time_series, targets = None):
        pass

    @abstractmethod
    def transform(self, time_series, targets = None):
        pass


class Compose(Transform):
    def __init__(self, transforms: list[BaseTransformer | Transform]):
        self.transforms = transforms
        for t in self.transforms:
            if not isinstance(t, Transform):
                assert hasattr(t, "fit") and callable(t.fit), "All transforms must have a fit method."
                assert hasattr(t, "transform") and callable(t.transform), "All transforms must have a transform method."

    # TODO: return new instead of self
    def __add__(self, other):
        self.transforms.append(other)
        return self
    
    def fit(self, ts: torch.Tensor, targets = None):
        for t in self.transforms:
            t.fit(ts, targets)
        self.is_fitted = True

    def transform(self, ts, targets = None):
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
