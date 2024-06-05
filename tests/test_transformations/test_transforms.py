import torch
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

time_series = torch.arange(0, 200, 1).repeat(200).reshape(200, 1, 200).float() / 100 - 1
time_series_unequal_length = time_series.clone().detach()
for i in range(200):
    time_series_unequal_length[i, 0, i + 1 :] = float("nan")

targets = torch.tensor([1, 2, 3, 4]).repeat(50)


def get_test_cases():
    return [
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
        PadBack(10, 5),
        PadFront(10),
        PadFront(10, 3),
        SlidingWindow(5),
        NaNToNumber(),
        NaNToNumber(5),
    ]


def test_normal_dataset():
    transforms = get_test_cases()

    print(time_series)
    for transform in transforms:
        transform.fit(time_series, targets)
        transformed_ts, transformed_tar = transform(time_series, targets)

        try:
            inv_transform = ~transform
            inv_ts, inv_tar = inv_transform(transformed_ts, transformed_tar)

            assert torch.allclose(time_series, inv_ts, atol=0.001)
            assert torch.allclose(targets, inv_tar, atol=0.001)
        except NoInverseError:
            pass


def test_not_equal_length_dataset():
    transforms = get_test_cases()

    for transform in transforms:
        transform.fit(time_series_unequal_length, targets)
        transformed_ts, transformed_tar = transform(time_series_unequal_length, targets)

        try:
            inv_transform = ~transform
            inv_ts, inv_tar = inv_transform(transformed_ts, transformed_tar)

            assert torch.allclose(time_series_unequal_length, inv_ts, atol=0.001, equal_nan=True)
            assert torch.allclose(targets, inv_tar, atol=0.001, equal_nan=True)
        except NoInverseError:
            pass
