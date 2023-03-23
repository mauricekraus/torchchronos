import os

from torchchronos.lightning import UCRUEAModule
from lightning import seed_everything


def test_lightning_gunpoint():
    seed_everything(12)
    mod = UCRUEAModule("GunPoint", split_ratio=(0.75, 0.15), batch_size=32)
    mod.prepare_data()
    mod.setup()

    batch = next(iter(mod.train_dataloader()))
    assert batch[0].shape == (150, 32, 1)
    assert batch[1].shape == (32, 1)
    assert mod.label_from_float_index(0.0) == "1"
