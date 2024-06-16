import pytest
import torch

from torchchronos.transforms.structure_transforms import Crop, Filter, PadBack, PadFront
from torchchronos.transforms.transformation_exceptions import NoInverseError


class TestCropTransform:
    def test_crop(self, test_case_with_target):
        time_series, target, has_unequal_length = test_case_with_target
        crop_transform = Crop(10, 20)
        crop_transform.fit(time_series)

        ts_cropped = crop_transform(time_series)
        assert torch.allclose(time_series[:, :, 10:20], ts_cropped, equal_nan=has_unequal_length)
        assert ts_cropped.shape == torch.Size([time_series.shape[0], time_series.shape[1], 10])

        if target is not None:
            ts_cropped, tar_cropped = crop_transform(time_series, target)
            assert torch.equal(target, tar_cropped)

    def test_crop_inverse(self):
        crop_transform = Crop(10, 20)

        with pytest.raises(NoInverseError):
            ~crop_transform


class TestPadBackTransform:
    def test_pad_back(self, test_case_with_target):
        time_series, target, has_unequal_length = test_case_with_target
        padback_transform = PadBack(10)
        padback_transform.fit(time_series)

        ts_padded = padback_transform(time_series)

        assert torch.allclose(
            time_series, ts_padded[:, :, : time_series.shape[2]], equal_nan=has_unequal_length
        )
        assert torch.equal(
            ts_padded[:, :, time_series.shape[2] :], torch.zeros((time_series.shape[0], 1, 10))
        )

        if target is not None:
            ts_padded, tar_padded = padback_transform(time_series, target)
            assert torch.equal(target, tar_padded)

    def test_pad_back_inverse(self, test_case_without_target):
        time_series, has_unequal_length = test_case_without_target

        padfront_transform = PadBack(10)
        padfront_transform.fit(time_series)
        inv_padback_transform = ~padfront_transform

        ts_padded = padfront_transform(time_series)
        ts_inv_padback = inv_padback_transform(ts_padded)

        assert torch.allclose(time_series, ts_inv_padback, equal_nan=has_unequal_length)


class TestPadFrontTransform:
    def test_pad_front(self, test_case_with_target):
        time_series, target, has_unequal_length = test_case_with_target
        padfront_transform = PadFront(10)
        padfront_transform.fit(time_series)

        ts_padded = padfront_transform(time_series)

        assert torch.allclose(time_series, ts_padded[:, :, 10:], equal_nan=has_unequal_length)
        assert torch.equal(ts_padded[:, :, :10], torch.zeros((time_series.shape[0], 1, 10)))

        if target is not None:
            ts_padded, tar_padded = padfront_transform(time_series, target)
            assert torch.equal(target, tar_padded)

    def test_pad_front_inverse(self, test_case_without_target):
        time_series, has_unequal_length = test_case_without_target

        padfront_transform = PadFront(10)
        padfront_transform.fit(time_series)
        inv_padfront_transform = ~padfront_transform

        ts_padded = padfront_transform(time_series)
        ts_inv_padfront = inv_padfront_transform(ts_padded)

        assert torch.allclose(time_series, ts_inv_padfront, equal_nan=has_unequal_length)


class TestFilterTransform:
    def test_filter(self, test_case_with_target):
        time_series, targets, has_unequal_length = test_case_with_target
        filter_transform = Filter(lambda x, y: y[0] == 0)

        if targets is not None:
            filter_transform.fit(time_series, targets)
            ts_filtered, tar_filtered = filter_transform(time_series, targets)

            filtered_indecies = torch.nonzero(torch.where(targets == 0, 1.0, 0.0), as_tuple=True)[0]

            assert torch.allclose(time_series[filtered_indecies], ts_filtered, equal_nan=has_unequal_length)
            assert torch.equal(torch.zeros((filtered_indecies.shape[0], 1)), tar_filtered)

    def test_filter_case_2(self, test_case_without_target):
        time_series, has_unequal_length = test_case_without_target
        filte_transform = Filter(lambda x, y: torch.max(x) == 3)
        filte_transform.fit(time_series)

        ts_filtered = filte_transform(time_series)

        filtered_indecies = torch.nonzero(torch.where(torch.max(time_series) == 3, 1.0, 0.0), as_tuple=True)[
            0
        ]
        assert torch.allclose(time_series[filtered_indecies], ts_filtered, equal_nan=has_unequal_length)

    def test_filter_inverse(self):
        filter_transform = Filter(lambda x, y: torch.max(x) == 3)

        with pytest.raises(NoInverseError):
            ~filter_transform
