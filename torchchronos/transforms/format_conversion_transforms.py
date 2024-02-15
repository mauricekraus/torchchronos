from typing import Optional

import torch
import numpy as np

from .base_transforms import Transform
from .transformation_exceptions import NoInverseError

"""
Change format of the data.
Examples are ToTorchTensor, and ToNumpyArray, change the datatype of the time_seies.
"""


class ToTorchTensor(Transform):
    def __init__(self):
        super().__init__(True)

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (torch.is_tensor(time_series) and targets is None) or (torch.is_tensor(time_series) and torch.is_tensor(targets)):
            return time_series, targets
        
        if targets is None:
            targets = None
        elif isinstance(targets[0], str):
            np_array = np.array(targets).astype(np.float32)
            targets = torch.from_numpy(np_array).float()
        else:
            targets = torch.tensor(targets).float()
        
        return torch.tensor(time_series).float(), targets

    def _invert(self) -> Transform:
        raise NoInverseError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToNumpyArray(Transform):
    def __init__(self):
        super().__init__(True)

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if targets is None:
            return time_series.numpy(), None
        else:
            return time_series.numpy(), targets.numpy()

    def _invert(self):
        raise NoInverseError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class To(Transform):
    def __init__(self, torch_attribute):
        super().__init__(True)
        self.torch_attribute = torch_attribute

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if targets is None:
            return time_series.to(self.torch_attribute), None
        else:
            return time_series.to(self.torch_attribute), targets.to(self.torch_attribute)

    def _invert(self):
        raise NoInverseError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attribute={self.torch_attribute})"
