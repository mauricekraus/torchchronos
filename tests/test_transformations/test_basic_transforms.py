import torch
import pytest

from torchchronos.transforms.basic_transforms import Identity, Scale, Shift, Normalize


def test_identity():
    transform = Identity()
    data = torch.randn(10, 1, 100)
    targets = torch.randint(0, 2, (100, 1))

    transformed_data = transform.transform(data)
    assert torch.equal(data, transformed_data)

    transformed_data, transformed_targets = transform.transform(data, targets)
    assert torch.equal(data, transformed_data)
    assert torch.equal(targets, transformed_targets)

    inverted_transform = transform.invert()
    assert isinstance(inverted_transform, Identity)
    assert str(inverted_transform) == "Identity()"

def test_scale():
    transform = Scale(torch.tensor([2.0]))
    data = torch.randn(10, 1, 100)
    targets = torch.randint(0, 2, (100, 1))

    transformed_data = transform.transform(data)
    assert torch.equal(data * 2.0, transformed_data)

    transformed_data, transformed_targets = transform.transform(data, targets)
    assert torch.equal(data * 2.0, transformed_data)
    assert torch.equal(targets, transformed_targets)

    inverted_transform = transform.invert()
    assert isinstance(inverted_transform, Scale)
    assert inverted_transform.scale == 0.5


    scale_tensor = torch.randint(1, 10, (100, 1)).float()
    transform = Scale(scale_tensor)
    transformed_data = transform.transform(data)
    assert torch.equal(data * scale_tensor, transformed_data)

def test_shift():
    transform = Shift(torch.tensor([2.0]))
    data = torch.randn(10, 1, 100)
    targets = torch.randint(0, 2, (100, 1))

    transformed_data = transform.transform(data)
    assert torch.equal(data + 2.0, transformed_data)

    transformed_data, transformed_targets = transform.transform(data, targets)
    assert torch.equal(data + 2.0, transformed_data)
    assert torch.equal(targets, transformed_targets)

    inverted_transform = transform.invert()
    assert isinstance(inverted_transform, Shift)
    assert inverted_transform.shift == -2.0

    shift_tensor = torch.randint(1, 10, (100, 1)).float()
    transform = Shift(shift_tensor)
    transformed_data = transform.transform(data)
    assert torch.equal(data + shift_tensor, transformed_data)

def test_normalize():
    transform = Normalize()

    data = torch.tensor([[1,2,3], [4,5,6]]).to(torch.float32).reshape(2, 1, 3)
    transform.fit(data)
    print(transform.std)
    assert torch.allclose(transform.mean, torch.tensor([2.5, 3.5, 4.5]).reshape(1, 3))
    assert torch.allclose(transform.std, torch.std(data, 0, True) + 1e-5)

    transformed_data = transform.transform(data)
    assert torch.allclose(transformed_data, (data - transform.mean) / transform.std)

    inverse_transform = transform.invert()
    inverted_data = inverse_transform.transform(transformed_data)
    assert torch.allclose(inverted_data, data)

    local_transformer = Normalize(local=True)
    local_transformed_data = local_transformer.transform(data)
    assert torch.allclose(local_transformed_data, (data - torch.mean(data, 2, True)) / (torch.std(data, 2, True) + 1e-5))
    