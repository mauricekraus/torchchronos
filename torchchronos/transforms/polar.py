import torch
from .base import Transform, Compose

# This class convertes a complex time series to polar coordinates
# Todo: implicit conversion to complex numbers?

class ComplexToPolar(Transform):

    def __init__(self, inverse=False):
        super().__init__(True)

    def _fit(self, time_series, targets=None) -> None:
        pass

    def _transform(self, time_series: torch.Tensor, targets=None) -> torch.Tensor:
        r = torch.abs(time_series)
        polar = torch.angle(time_series)

        ts_stacked = torch.stack((r, polar), dim=1).squeeze()
        return ts_stacked.type(torch.float32), targets
        
    def _invert(self):
        return PolarToComplex()
    
    def __repr__(self) -> str:
        return f"{__class__.__name__}()"
    
# This class convertes a polar time series to complex numbers
class PolarToComplex(Transform):
    def __init__(self):
        super().__init__(True)

    def _fit(self, time_series, targets=None) -> None:
        pass

    def _transform(self, time_series: torch.Tensor, targets=None) -> torch.Tensor:
        r = time_series[:, 0, :]
        polar = time_series[:, 1, :]

        ts_stacked = r * torch.exp(1j * polar)

        return ts_stacked.type(torch.complex128), targets
    
    def _invert(self):
        return ComplexToPolar()
    
    def __repr__(self) -> str:
        return f"{__class__.__name__}()"