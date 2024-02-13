import torch

from .base_transforms import Transform

"""
Change format of the data.
Examples are ToTorchTensor, and ToNumpyArray, change the datatype of the time_seies.
"""


class ToTorchTensor(Transform):
    def __init__(self):
        super().__init__(True)

    def _fit(self, time_series, targets=None) -> None:
        pass

    def _transform(self, time_series: torch.Tensor, targets=None) -> torch.Tensor:
        targets = None if targets is None else torch.tensor(targets)
        if torch.is_tensor(time_series):
            return time_series, targets
        return torch.tensor(time_series).float(), targets

    def _invert(self):
        return self  # TODO: maybe raise error

    def __repr__(self) -> str:
        return "ToTorchTensor()"


class ToNumpyArray(Transform):
    def __init__(self):
        super().__init__(True)

    def _fit(self, time_series, targets=None) -> None:
        pass

    def _transform(self, time_series: torch.Tensor, targets=None) -> torch.Tensor:
        raise NotImplementedError("Not yet implemented")

    def _invert(self):
        return self  # TODO: maybe raise error

    def __repr__(self) -> str:
        return f"{__class__.__name__}()"

# TODO: change to To und dann entweder device oder dtype, wie bei torch.to(...)
class ChangeDataType(Transform):
    def __init__(self, dtype):
        super().__init__(True)
        # TODO: chech if dtype is valid and in available in pytorch
        self.dtype = dtype

    def _fit(self, time_series, targets=None) -> None:
        pass

    def _transform(self, time_series: torch.Tensor, targets=None) -> torch.Tensor:
        return time_series.type(self.dtype), targets

    def _invert(self):
        return self  # TODO: maybe raise error

    def __repr__(self) -> str:
        return f"{__class__.__name__}(dtype={self.dtype})"
