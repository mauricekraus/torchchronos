# Pytest of download file
import os

import pytest

from torchchronos.lightning import UCRUEAModule
from lightning import seed_everything

from torchchronos.transforms import Compose, PadFront, PadBack


def test_padfront_gunpoint():
    seed_everything(12)
    mod = UCRUEAModule(
        "GunPoint",
        split_ratio=(0.75, 0.15),
        batch_size=32,
        transform=Compose([PadFront(10)]),
    )
    mod.prepare_data()
    mod.setup()

    batch = next(iter(mod.train_dataloader()))
    assert batch[0].shape == (160, 32, 1)
    assert batch[0][:10].sum() == 0


def test_padback_gunpoint():
    seed_everything(12)
    mod = UCRUEAModule(
        "GunPoint",
        split_ratio=(0.75, 0.15),
        batch_size=32,
        transform=Compose([PadBack(10)]),
    )
    mod.prepare_data()
    mod.setup()

    batch = next(iter(mod.train_dataloader()))
    assert batch[0].shape == (160, 32, 1)
    assert batch[0][-10:].sum() == 0


def test_pad_gunpoint():
    seed_everything(12)
    mod = UCRUEAModule(
        "GunPoint",
        split_ratio=(0.75, 0.15),
        batch_size=32,
        transform=Compose([PadBack(10), PadFront(10)]),
    )
    mod.prepare_data()
    mod.setup()

    batch = next(iter(mod.train_dataloader()))
    assert batch[0].shape == (170, 32, 1)
    assert batch[0][-10:].sum() == 0
    assert batch[0][:10].sum() == 0


def test_properties_gunpoint():
    seed_everything(12)
    mod = UCRUEAModule(
        "GunPoint",
        split_ratio=(0.75, 0.15),
        batch_size=32,
        transform=Compose([PadBack(10), PadFront(10)]),
    )

    with pytest.raises(Exception) as e_info:
        mod.num_classes

    mod.prepare_data()
    mod.setup()

    assert mod.num_classes == 2
    assert mod.dimensions == 1
    assert mod.series_length == 150
    assert mod.equal_length
    assert mod.univariate
