import os

from torchchronos.lightning import UCRUEAModule
from lightning import seed_everything


def test_lightning_phoneme_spectra():
    seed_everything(12)
    os.system("rm -rf .cache/data/PhonemeSpectra")
    mod = UCRUEAModule("PhonemeSpectra", split_ratio=(0.75, 0.15), batch_size=32)
    mod.prepare_data()
    mod.setup()

    batch = next(iter(mod.train_dataloader()))
    assert batch[0].shape == (217, 32, 11)
    assert batch[1].shape == (32,)
    assert mod.label_from_index(0.0) == "aa"
    assert mod.label_from_index(1.0) == "ae"
