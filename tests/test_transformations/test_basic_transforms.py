import torch

from torchchronos.transforms.basic_transforms import Identity, Normalize, Scale, Shift


class TestIdentityTransform:
    def test_case_1(self, test_case_with_target):
        time_series, targets, has_unequal_length = test_case_with_target
        transform = Identity()

        ts_transformed = transform(time_series)
        assert torch.allclose(time_series, ts_transformed, equal_nan=has_unequal_length)

        if targets is not None:
            ts_transformed, tar_transformed = transform(time_series, targets)
            assert torch.allclose(time_series, ts_transformed, equal_nan=has_unequal_length)
            assert torch.equal(targets, tar_transformed)

    def test_inverse(self):
        transform = Identity()
        inv_transform = ~transform
        assert isinstance(inv_transform, Identity)


class TestScaleTransform:
    def test_scale(self, test_case_with_target):
        time_series, targets, has_unequal_length = test_case_with_target
        transform = Scale(torch.tensor([2.0]))

        transformed_data = transform(time_series)
        assert torch.allclose(time_series * 2.0, transformed_data, equal_nan=has_unequal_length)

        if targets is not None:
            transformed_data, transformed_targets = transform(time_series, targets)
            assert torch.allclose(time_series * 2.0, transformed_data, equal_nan=has_unequal_length)
            assert torch.equal(targets, transformed_targets)

    def test_scale_invert(self, test_case_without_target):
        time_series, has_unequal_length = test_case_without_target
        transform = Scale(2.0)
        inv_transform = transform.invert()
        assert isinstance(inv_transform, Scale)
        assert inv_transform.scale == 0.5

        ts_transformed = transform(time_series)
        ts_inv_transformed = inv_transform(ts_transformed)
        assert torch.allclose(time_series, ts_inv_transformed, equal_nan=has_unequal_length)

    def test_scale_with_vector(self, test_case_without_target):
        time_series, has_unequal_length = test_case_without_target
        n_time_steps = time_series.shape[2]
        scale_tensor = torch.randint(1, 10, (n_time_steps, 1)).float()
        transform = Scale(scale_tensor)

        transformed_data = transform.transform(time_series)
        assert torch.allclose(transformed_data, time_series * scale_tensor, equal_nan=has_unequal_length)


class TestShiftTransform:
    def test_shift(self, test_case_with_target):
        time_series, targets, has_unequal_length = test_case_with_target
        transform = Shift(torch.tensor([2.0]))

        transformed_data = transform(time_series)
        assert torch.allclose(time_series + 2.0, transformed_data, equal_nan=has_unequal_length)

        if targets is not None:
            transformed_data, transformed_targets = transform(time_series, targets)
            assert torch.allclose(time_series + 2.0, transformed_data, equal_nan=has_unequal_length)
            assert torch.equal(targets, transformed_targets)

    def test_shift_inverse(self, test_case_without_target):
        time_series, has_unequal_length = test_case_without_target
        transform = Shift(2.0)
        inv_tranform = transform.invert()
        assert isinstance(inv_tranform, Shift)
        assert inv_tranform.shift == -2.0

        ts_transformed = transform(time_series)
        inv_ts_transformed = inv_tranform(ts_transformed)
        assert torch.allclose(time_series, inv_ts_transformed, equal_nan=has_unequal_length, atol=1e-5)

    def test_shift_with_vector(self, test_case_without_target):
        time_series, has_unequal_length = test_case_without_target
        n_time_steps = time_series.shape[2]
        shift_tensor = torch.randint(1, 10, (n_time_steps, 1)).float()
        transform = Shift(shift_tensor)

        ts_transformed = transform(time_series)
        assert torch.allclose(time_series + shift_tensor, ts_transformed, equal_nan=has_unequal_length)


class TestNormalizeTransform:
    def test_normalize(self, test_case_with_target):
        time_series, target, has_unequal_length = test_case_with_target
        transform = Normalize()

        transform.fit(time_series)
        ts_transformed = transform(time_series)
        if not has_unequal_length:
            # Normal Case
            assert torch.allclose(
                torch.mean(ts_transformed, dim=[0], keepdim=True),
                torch.zeros_like(torch.mean(ts_transformed, dim=[0], keepdim=True)),
                atol=1e-5,
            )
            assert torch.allclose(
                torch.std(ts_transformed, dim=0, keepdim=True),
                torch.ones_like(torch.std(ts_transformed, dim=0, keepdim=True)),
                atol=1e-3,
            )
        else:
            # Case with nones
            pass

    def test_inverse(self, test_case_without_target):
        time_series, has_unequal_length = test_case_without_target
        transform = Normalize()
        transform.fit(time_series)
        inv_transform = ~transform

        ts_transformed = transform(time_series)
        inv_ts_transformed = inv_transform(ts_transformed)
        print(time_series - inv_ts_transformed)
        assert torch.allclose(time_series, inv_ts_transformed, equal_nan=has_unequal_length, atol=1e-5)

    def test_local_normalize(self, test_case_without_target):
        time_series, has_unequal_length = test_case_without_target
        transform = Normalize(local=True)
        ts_transformed = transform(time_series)

        if not has_unequal_length:
            ts_transformed
            # mean = torch.mean(ts_transformed, dim=2)
            # res = torch.allclose(
            #     torch.mean(ts_transformed, dim=1, keepdim=True),
            #     torch.zeros_like(torch.mean(ts_transformed, dim=1, keepdim=True)),
            #     atol=1e-5,
            # )
        # print(res)
