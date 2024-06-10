import pytest
import torch
from torch.utils.data import TensorDataset
from torchchronos.transforms import (
    CombineToComplex,
    Crop,
    Filter,
    Identity,
    LabelTransform,
    NaNToNumber,
    NoInverseError,
    Normalize,
    PadBack,
    PadFront,
    Scale,
    Shift,
    SlidingWindow,
    To,
    ToNumpyArray,
    ToTorchTensor,
)


@pytest.fixture
def test_time_series():
    return torch.arange(0, 200, 1).repeat(200).reshape(200, 1, 200).float() / 100 - 1


@pytest.fixture
def test_time_series_unequal_length(test_time_series):
    test_ts_unequal_length = test_time_series.clone().detach()
    for i in range(200):
        test_ts_unequal_length[i, 0, i + 1 :] = float("nan")
    return test_ts_unequal_length


@pytest.fixture
def targets():
    return torch.tensor([1, 2, 3, 4]).repeat(50)


@pytest.fixture
def test_tensor_dataset(test_time_series, targets):
    return TensorDataset(test_time_series, targets)


test_cases = [
    LabelTransform(),
    CombineToComplex(),
    ToTorchTensor(),
    ToNumpyArray(),
    To(torch.float16),
    Normalize(),
    # Normalize(local=True),
    Identity(),
    Scale(4),
    Scale(torch.arange(1, 201, 1) / 50),
    Shift(2.5),
    Shift(torch.arange(1, 201, 1) / 50),
    Filter(lambda ts, tar: tar == 1),
    Crop(0, 10),
    PadBack(10),
    PadBack(length=10, value=5),
    PadFront(10),
    PadFront(length=10, value=3),
    SlidingWindow(5),
    NaNToNumber(),
    NaNToNumber(5),
]


@pytest.mark.parametrize("transform", test_cases)
def test_equal_length(transform, test_time_series, targets):
    transform.fit(test_time_series, targets)
    transformed_dataset, transformed_targets = transform(test_time_series, targets)

    try:
        inv_transform = ~transform
        inv_dataset, inv_targets = inv_transform(transformed_dataset, transformed_targets)

        assert torch.allclose(test_time_series, inv_dataset, atol=0.001), (
            f"Inverse transform failed for {transform} on time_series. "
            f"Expected: {test_time_series},"
            f"Got: {inv_dataset}"
        )
        assert torch.allclose(targets, inv_targets, atol=0.001), (
            f"Inverse transform failed for {transform} on targets. "
            f"Expected: {targets}, Got: {inv_targets}"
        )
    except NoInverseError:
        pass


@pytest.mark.parametrize("transform", test_cases)
def test_unequal_length(transform, test_time_series_unequal_length, targets):
    transform.fit(test_time_series_unequal_length, targets)
    transformed_dataset, transformed_targets = transform(test_time_series_unequal_length, targets)

    try:
        inv_transform = ~transform
        inv_dataset, inv_targets = inv_transform(transformed_dataset, transformed_targets)

        assert torch.allclose(test_time_series_unequal_length, inv_dataset, atol=0.001, equal_nan=True), (
            f"Inverse transform failed for {transform} on time_series. "
            f"Expected: {test_time_series_unequal_length},"
            f"Got: {inv_dataset}"
        )
        assert torch.allclose(targets, inv_targets, atol=0.001, equal_nan=True), (
            f"Inverse transform failed for {transform} on targets. "
            f"Expected: {targets}, Got: {inv_targets}"
        )
    except NoInverseError:
        pass


@pytest.mark.parametrize("transform", test_cases)
def test_dataset(transform, test_tensor_dataset):
    transform.fit(test_tensor_dataset)

    try:
        ~transform

    except NoInverseError:
        pass
