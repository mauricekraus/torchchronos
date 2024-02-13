import torch

from torchchronos.transforms.spectral_transforms import FourierTransform, InverseFourierTransform


def test_fourier_transform():
    transform = FourierTransform()
    time_series = torch.tensor([1, 2, 3, 4, 5])
    ft = transform.transform(time_series)

    assert isinstance(ft, torch.Tensor)
    assert ft.shape == time_series.shape

    inverse_transform = ~transform

    assert isinstance(inverse_transform, InverseFourierTransform)
    assert repr(transform) == "FourierTransform()"

def test_inverse_fourier_transform():
    transform = InverseFourierTransform()
    time_series = torch.tensor([1, 2, 3, 4, 5])
    ifft = transform.transform(time_series)

    assert isinstance(ifft, torch.Tensor)
    assert ifft.shape == time_series.shape

    inverse_transform = ~transform

    assert isinstance(inverse_transform, FourierTransform)
    assert repr(transform) == "InverseFourierTransform()"
