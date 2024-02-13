import pytest
import torch

from torchchronos.transforms.structure_transforms import Crop, Filter, PadBack, PadFront, Split
from torchchronos.transforms.transformation_exceptions import NoInverseError


def test_crop():
    data = torch.randn((10, 1, 100), dtype=torch.float32)

    crop = Crop(10, 20)

    crop.fit(data)
    cropped_data = crop(data)

    assert cropped_data.shape == torch.Size([10, 1, 10])
    assert torch.allclose(cropped_data, data[:, :, 10:20])


def test_pad_back():
    data = torch.randn((10, 1, 10), dtype=torch.float32)

    pad = PadBack(10)

    pad.fit(data)
    padded_data = pad(data)

    assert padded_data.shape == torch.Size([10, 1, 20])
    assert torch.allclose(padded_data[:, :, :10], data)
    assert torch.allclose(padded_data[:, :, 10:], torch.zeros((10, 1, 10)))


def test_pad_front():
    data = torch.randn((10, 1, 10), dtype=torch.float32)

    pad = PadFront(10)

    pad.fit(data)
    padded_data = pad(data)

    assert padded_data.shape == torch.Size([10, 1, 20])
    assert torch.allclose(padded_data[:, :, 10:], data)
    assert torch.allclose(padded_data[:, :, :10], torch.zeros((10, 1, 10)))


def test_crop_inverse():
    crop = Crop(10, 20)

    with pytest.raises(NoInverseError):
        ~crop


def test_pad_front_inverse():
    data = torch.randn((10, 1, 10), dtype=torch.float32)

    pad = PadFront(10)

    pad.fit(data)
    padded_data = pad(data)
    print(padded_data.shape, pad.time_series_length)

    inverse = ~pad
    inverse_data = inverse(padded_data)

    assert isinstance(inverse, Crop)
    assert inverse.start == 10
    assert inverse.end == 20
    assert inverse_data.shape == torch.Size([10, 1, 10])
    assert torch.allclose(inverse_data, data)


def test_pad_back_inverse():
    data = torch.randn((10, 1, 10), dtype=torch.float32)

    pad = PadBack(10)

    pad.fit(data)
    padded_data = pad(data)

    inverse = ~pad
    inverse_data = inverse(padded_data)

    assert isinstance(inverse, Crop)
    assert inverse.start == 0
    assert inverse.end == 10
    assert inverse_data.shape == torch.Size([10, 1, 10])
    assert torch.allclose(inverse_data, data)


def test_filter():
    data = torch.randn((10, 1, 10), dtype=torch.float32)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32).view(
        -1, 1
    )

    filter = Filter(lambda x, y: y[0] == 0)

    filter.fit(data, targets)
    filtered_data, filtered_targets = filter(data, targets)

    print(filtered_data.shape, filtered_targets.shape)
    assert filtered_data.shape == torch.Size([5, 1, 10])
    assert filtered_targets.shape == torch.Size([5, 1])
    assert torch.allclose(filtered_data, data[targets[:, 0] == 0])
    assert torch.allclose(filtered_targets, targets[targets[:, 0] == 0])

    data = torch.tensor(
        [[1, 0, 3], [1, 1, 1], [2, 4, 3], [3, 0, 0]], dtype=torch.float32
    ).reshape(4, 1, 3)

    filter = Filter(lambda x, y: torch.max(x) == 3)

    filter.fit(data)
    filtered_data = filter(data)

    assert filtered_data.shape == torch.Size([2, 1, 3])


    
