import pytest
import torch

from tests.test_transformations.test_utils import test_cases
from torchchronos.transforms.basic_transforms import Identity, Normalize, Scale, Shift


@pytest.mark.parametrize("time_series, targets, has_unequal_length", [*test_cases])
class TestIdentityTransform:
    def test_case_1(self, time_series, targets, has_unequal_length):
        transform = Identity()

        ts_transformed = transform(time_series)
        assert torch.allclose(time_series, ts_transformed, equal_nan=has_unequal_length)

        if targets is not None:
            ts_transformed, tar_transformed = transform(time_series, targets)
            assert torch.allclose(time_series, ts_transformed, equal_nan=has_unequal_length)
            assert torch.equal(targets, tar_transformed)

    def test_inverse(self, time_series, targets, has_unequal_length):
        transform = Identity()
        inv_transform = ~transform
        assert isinstance(inv_transform, Identity)


@pytest.mark.parametrize("time_series, targets, has_unequal_length", [*test_cases])
class TestScaleTransform:
    def test_scale(self, time_series, targets, has_unequal_length):
        transform = Scale(torch.tensor([2.0]))

        transformed_data = transform(time_series)
        assert torch.allclose(time_series * 2.0, transformed_data, equal_nan=has_unequal_length)

        if targets is not None:
            transformed_data, transformed_targets = transform(time_series, targets)
            assert torch.allclose(time_series * 2.0, transformed_data, equal_nan=has_unequal_length)
            assert torch.equal(targets, transformed_targets)

    def test_scale_invert(self, time_series, targets, has_unequal_length):
        transform = Scale(2.0)
        inv_transform = transform.invert()
        assert isinstance(inv_transform, Scale)
        assert inv_transform.scale == 0.5

        ts_transformed = transform(time_series)
        ts_inv_transformed = inv_transform(ts_transformed)
        assert torch.allclose(time_series, ts_inv_transformed, equal_nan=has_unequal_length)

    def test_scale_with_vector(self, time_series, targets, has_unequal_length):
        n_time_steps = time_series.shape[2]
        scale_tensor = torch.randint(1, 10, (n_time_steps, 1)).float()
        transform = Scale(scale_tensor)

        transformed_data = transform.transform(time_series)
        assert torch.allclose(transformed_data, time_series * scale_tensor, equal_nan=has_unequal_length)


@pytest.mark.parametrize("time_series, targets, has_unequal_length", [*test_cases])
class TestShiftTransform:
    def test_shift(self, time_series, targets, has_unequal_length):
        transform = Shift(torch.tensor([2.0]))

        transformed_data = transform(time_series)
        assert torch.allclose(time_series + 2.0, transformed_data, equal_nan=has_unequal_length)

        if targets is not None:
            transformed_data, transformed_targets = transform(time_series, targets)
            assert torch.allclose(time_series + 2.0, transformed_data, equal_nan=has_unequal_length)
            assert torch.equal(targets, transformed_targets)

    def test_shift_inverse(self, time_series, targets, has_unequal_length):
        transform = Shift(2.0)
        inv_tranform = transform.invert()
        assert isinstance(inv_tranform, Shift)
        assert inv_tranform.shift == -2.0

        ts_transformed = transform(time_series)
        inv_ts_transformed = inv_tranform(ts_transformed)
        assert torch.allclose(time_series, inv_ts_transformed, equal_nan=has_unequal_length)

    def test_shift_with_vector(self, time_series, targets, has_unequal_length):
        n_time_steps = time_series.shape[2]
        shift_tensor = torch.randint(1, 10, (n_time_steps, 1)).float()
        transform = Shift(shift_tensor)

        ts_transformed = transform(time_series)
        assert torch.allclose(time_series + shift_tensor, ts_transformed, equal_nan=has_unequal_length)


@pytest.mark.parametrize("time_series, targets, has_unequal_length", [*test_cases])
class TestNormalizeTransform:
    def test_normalize(self, time_series, targets, has_unequal_length):
        transform = Normalize()

        transform.fit(time_series)
        # nan_mean = np.nanmean(time_series)
        # assert torch.allclose(nan_mean, transform.mean)
        # assert torch.allclose(transform.std, torch.std(data, 0, True) + 1e-5)

        # transformed_data = transform.transform(data)
        # assert torch.allclose(transformed_data, (data - transform.mean) / transform.std)

        # inverse_transform = transform.invert()
        # inverted_data = inverse_transform.transform(transformed_data)
        # assert torch.allclose(inverted_data, data)

        # local_transformer = Normalize(local=True)
        # local_transformed_data = local_transformer.transform(data)
        # assert torch.allclose(
        #     local_transformed_data, (data - torch.mean(data, 2, True)) / (torch.std(data, 2, True) + 1e-5)
        # )
