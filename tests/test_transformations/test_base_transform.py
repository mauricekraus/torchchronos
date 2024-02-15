import torch
import pytest
import tempfile
from pathlib import Path

from torchchronos.transforms.base_transforms import Transform, Compose
from torchchronos.transforms.format_conversion_transforms import ToTorchTensor
from torchchronos.transforms.structure_transforms import SlidingWindow, RemoveLabels
from torchchronos.transforms.basic_transforms import Shift
from torchchronos.datasets.util.aeon_datasets import AeonClassificationDataset
from torchchronos.datasets.util.base_dataset import BaseDataset

transform = Shift(5)
rm_labels = RemoveLabels()
data = torch.randn(10, 1, 100)
targets = torch.randint(0, 2, (100, 1))
dataset = BaseDataset(data, targets)


def test_transform_transform():

    transform.fit(data)

    transformed_data = transform.transform(data)
    assert type(transformed_data) == torch.Tensor

    transformed_data = transform.transform(data, None)
    assert type(transformed_data) == torch.Tensor

    transformed_data, transformed_targets = transform.transform(data, targets)
    assert type(transformed_data) == torch.Tensor
    assert type(transformed_targets) == torch.Tensor

    #check if targets is none after transform, that nothing is returned
    transformed_data = rm_labels.transform(data, targets)
    assert type(transformed_data) == torch.Tensor

    dataset_transformed = transform.transform(dataset)
    assert type(dataset_transformed) == BaseDataset

    dataset_transformed = transform.transform(dataset, None)
    assert type(dataset_transformed) == BaseDataset

    with pytest.raises(Exception):
        dataset_transformed = transform.transform(dataset, targets)

def test_transform_call():
    transform = Shift(5)
    rm_labels = RemoveLabels()
    data = torch.randn(10, 1, 100)
    targets = torch.randint(0, 2, (100, 1))
    dataset = AeonClassificationDataset(name="GunPoint")
    transform.fit(data)

    transformed_data = transform(data)
    assert type(transformed_data) == torch.Tensor

    transformed_data = transform(data, None)
    assert type(transformed_data) == torch.Tensor

    transformed_data, transformed_targets = transform(data, targets)
    assert type(transformed_data) == torch.Tensor
    assert type(transformed_targets) == torch.Tensor

    #check if targets is none after transform, that nothing is returned
    transformed_data = rm_labels(data, targets)
    assert type(transformed_data) == torch.Tensor

    dataset_transformed = transform(dataset)
    assert type(dataset_transformed) == BaseDataset

    dataset_transformed = transform(dataset, None)
    assert type(dataset_transformed) == BaseDataset

    with pytest.raises(Exception):
        dataset_transformed = transform(dataset, targets)

def test_fit():
    return_type = transform.fit(data)
    assert return_type == None

    return_type = transform.fit(data, None)
    assert return_type == None

    return_type = transform.fit(dataset)
    assert return_type == None

    return_type = transform.fit(dataset, None)
    assert return_type == None

    return_type = transform.fit(data, targets)
    assert return_type == None

def test_fit_transform():
    transformed_data = transform.fit_transform(data)
    assert type(transformed_data) == torch.Tensor

    transformed_data = transform.fit_transform(data, None)
    assert type(transformed_data) == torch.Tensor

    transformed_data = transform.fit_transform(dataset)
    assert type(transformed_data) == BaseDataset

    transformed_data = transform.fit_transform(dataset, None)
    assert type(transformed_data) == BaseDataset

    transformed_data, transformed_targets = transform.fit_transform(data, targets)
    assert type(transformed_data) == torch.Tensor
    assert type(transformed_targets) == torch.Tensor


def test_add():
    new_transform = transform + rm_labels
    assert type(new_transform) == Compose
    assert len(new_transform.transforms) == 2

def test_invert():
    assert transform._invert_transform == None

    inverted_transform = ~transform
    assert inverted_transform._invert_transform == transform

    inverted_transform = transform.invert()
    assert inverted_transform._invert_transform == transform

    assert transform._invert_transform == inverted_transform

def test_save_load():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        transform.save("test_transformation", path=temp_path)

        loaded_transform = Transform.load("test_transformation", path=temp_path)

        assert isinstance(loaded_transform, Shift)
        assert loaded_transform.shift == 5
        assert loaded_transform.is_fitted == True  
    


def test_compose_init():
    transform = Compose([Shift(5), RemoveLabels()])
    assert len(transform.transforms) == 2

def test_compose_add():
    transform = Compose([Shift(5)])
    new_transform = transform + RemoveLabels()
    
    assert transform == transform
    assert type(new_transform) == Compose
    assert len(new_transform.transforms) == 2

def test_compose_fit():
    shift = Shift(5)
    shift.is_fitted = False
    transform = Compose([shift, shift])
    transform.fit(data)

    assert shift.is_fitted == True


def test_compose_transform():
    transform = Compose([Shift(5), Shift(15)])
    transform.fit(data)
    transformed_data = transform.transform(data)
    assert type(transformed_data) == torch.Tensor
    assert torch.isclose(transformed_data, data + 20).all()

def test_compose_invert():
    transform = Compose([Shift(5), Shift(15)])
    inverted_transform = ~transform
    assert type(inverted_transform) == Compose
    assert len(inverted_transform.transforms) == 2
    assert inverted_transform.transforms[0].shift == -15
    assert inverted_transform.transforms[1].shift == -5

    
def test_compose_example():

    transform = Compose([ToTorchTensor(), SlidingWindow(10, 3), Shift(shift=1)])
    transform.fit(dataset.data, dataset.targets)
    transformed_dataset = transform(dataset)

    dataset = AeonClassificationDataset(name="GunPoint")
    dataset.prepare()
    dataset.load()

    transform = Compose([ToTorchTensor(), RemoveLabels(), SlidingWindow(10, 3), Shift(shift=1)])
    transform.fit(dataset.data, dataset.targets)
    transformed_dataset = transform(dataset)
