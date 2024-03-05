import pytest
import torch

from torchchronos.datasets.prepareable_dataset import PrepareableDataset, NotPreparedError, NotLoadedError
from torchchronos.datasets.aeon_datasets import AeonClassificationDataset
from torchchronos.transforms.basic_transforms import Shift


def test_prepareable_dataset_exceptions():
    dataset = AeonClassificationDataset(name="GunPoint")

    with pytest.raises(NotPreparedError):
        dataset[0]

    with pytest.raises(NotPreparedError):
        dataset.load()

    dataset.prepare()

    with pytest.raises(NotLoadedError):
        dataset[0]

    dataset.load()


def test_prepare():
    dataset = AeonClassificationDataset(name="GunPoint")
    assert dataset.is_prepared is False
    dataset.prepare()
    assert dataset.is_prepared is True


def test_load():
    dataset = AeonClassificationDataset(name="GunPoint")
    assert dataset.is_loaded is False
    dataset.prepare()
    dataset.load()
    assert dataset.is_loaded is True


def test_transforms():
    dataset = AeonClassificationDataset(name="GunPoint")
    dataset.prepare()
    dataset.load()

    transform = Shift(5)
    shifted_dataset = AeonClassificationDataset(name="GunPoint", transform=transform)
    shifted_dataset.prepare()
    shifted_dataset.load()

    assert torch.equal(shifted_dataset[:][0], dataset[:][0] + 5)
