# Pytest of download file
import os

from torchchronos.lightning import UCRUEAModule
from lightning import seed_everything

from torchchronos.transforms import Compose, PadFront, PadBack


def test_padfront_gunpoint():
    seed_everything(12)
    os.system("rm -rf .cache/data/GunPoint")
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
    os.system("rm -rf .cache/data/GunPoint")
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
    os.system("rm -rf .cache/data/GunPoint")
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
