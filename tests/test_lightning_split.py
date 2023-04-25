from math import ceil
import os
from tests.data_class import Dataset

from torchchronos.lightning import UCRUEADataModule
from lightning import seed_everything
import pytest


@pytest.fixture
def dataset_fixture():
    return [
        Dataset("NATOPS", 180, 180),
        Dataset("PhonemeSpectra", 3315, 3353),
        Dataset("NonInvasiveFetalECGThorax2", 1800, 1965),
    ]


def test_lightning_custom_split_float_number(dataset_fixture: list[Dataset]):
    seed_everything(12)

    for dataset in dataset_fixture:
        split_ratio = (0.75, 0.15)
        batch_size = 32
        mod = UCRUEADataModule(
            dataset.name, split_ratio=split_ratio, batch_size=batch_size
        )
        mod.prepare_data()
        mod.setup()

        total = dataset.total / batch_size

        train_split_len = ceil(split_ratio[0] * total)
        val_split_len = ceil(split_ratio[1] * total)
        test_split_len = ceil(round(1 - split_ratio[0] - split_ratio[1], 2) * total)

        assert len(mod.train_dataloader()) == train_split_len
        assert len(mod.val_dataloader()) == val_split_len
        assert len(mod.test_dataloader()) == test_split_len


def test_lightning_original_split_float_number(dataset_fixture: list[Dataset]):
    seed_everything(12)
    for dataset in dataset_fixture:
        split_ratio = 0.75
        batch_size = 32
        mod = UCRUEADataModule(
            dataset.name, split_ratio=split_ratio, batch_size=batch_size
        )
        mod.prepare_data()
        mod.setup()

        train = dataset.train / batch_size

        train_split_len = ceil(split_ratio * train)
        val_split_len = ceil(round(1 - split_ratio, 2) * train)
        test_split_len = ceil(dataset.test / batch_size)

        assert len(mod.train_dataloader()) == train_split_len
        assert len(mod.val_dataloader()) == val_split_len
        assert len(mod.test_dataloader()) == test_split_len
