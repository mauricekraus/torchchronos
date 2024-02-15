import torch
import numpy as np
import pytest

from torchchronos.transforms.transformation_exceptions import NoInverseError
from torchchronos.transforms.format_conversion_transforms import ToTorchTensor, ToNumpyArray, To




def test_ToTorchTensor():
    transform = ToTorchTensor()
    numpy_data = np.random.rand(10, 1, 10)
    numpy_targets = np.random.rand(10, 1)
    numpy_string_targets = np.array(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

    torch_data = transform(numpy_data)
    assert isinstance(torch_data, torch.Tensor)

    torch_data, torch_targets = transform(numpy_data, numpy_targets)
    assert isinstance(torch_data, torch.Tensor)
    assert isinstance(torch_targets, torch.Tensor)

    torch_data, torch_targets = transform(numpy_data, numpy_string_targets)
    assert isinstance(torch_data, torch.Tensor)
    assert isinstance(torch_targets, torch.Tensor)

    with pytest.raises(NoInverseError):
        transform.invert()

def test_ToNumpyArray():
    transform = ToNumpyArray()
    torch_data = torch.rand(10, 1, 10)
    torch_targets = torch.rand(10, 1)

    numpy_data = transform(torch_data)
    assert isinstance(numpy_data, np.ndarray)

    numpy_data, numpy_targets = transform(torch_data, torch_targets)
    assert isinstance(numpy_data, np.ndarray)
    assert isinstance(numpy_targets, np.ndarray)

    with pytest.raises(NoInverseError):
        transform.invert()


def test_To():
    transform = To(torch.float32)
    data = torch.randint(0, 10, (10, 1, 10))
    targets = torch.randint(-4, 8, (10, 1))

    transformed_data, transformed_targets = transform(data, targets)
    assert transformed_data.dtype == torch.float32
    assert transformed_targets.dtype == torch.float32
    assert transformed_data.shape == data.shape

    with pytest.raises(NoInverseError):
        transform.invert()

