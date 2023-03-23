import torch


class Transform:
    def fit(self, ts: torch.Tensor) -> "Transform":
        return self

    def __call__(self, ts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def fit(self, ts: torch.Tensor) -> Transform:
        for t in self.transforms:
            t.fit(ts)
        return self

    def __call__(self, ts):
        for t in self.transforms:
            ts = t(ts)
        return ts

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
