from typing import Optional
import torch

from .base_transforms import Transform

class FourierTransform(Transform):  # TODO: add kwargs for fft
    def __init__(self) -> None:
        super().__init__(True)

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        ft = torch.fft.fft(time_series, norm="ortho")
        return ft, targets

    def _invert(self):
        return InverseFourierTransform()

    def __repr__(self) -> str:
        return "FourierTransform()"


class InverseFourierTransform(Transform):
    def __init__(self):
        super().__init__(True)

    def _fit(
        self, time_series:torch.Tensor, targets:Optional[torch.Tensor]= None
        ) -> None:
        pass

    def _transform(
        self, time_series: torch.Tensor, targets:Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        ifft = torch.fft.ifft(time_series, norm="ortho")
        return ifft, targets

    def _invert(self):
        return FourierTransform()

    def __repr__(self) -> str:
        return "InverseFourierTransform()"