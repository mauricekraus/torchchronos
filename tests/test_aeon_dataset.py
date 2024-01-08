from pathlib import Path
import os
from shutil import rmtree
from torchchronos.datasets.util.aeon_datasets import AeonDataset
from aeon.datasets._data_loaders import load_classification



def test_aeon_dataset_custom_path():
    dataset_name = "GunPoint"
    torch_chronos_path = Path(".cache/data/torchchronos/tests")
    aeon_path = Path(".cache/data/aeon/tests")
    if torch_chronos_path.exists():
        rmtree(torch_chronos_path)
    assert not torch_chronos_path.exists()

    dataset = AeonDataset(dataset_name, save_path=torch_chronos_path)

    assert dataset.name == dataset_name

    dataset.prepare()

    assert dataset.is_prepared
    assert (torch_chronos_path / "GunPoint").exists()

    X, y = load_classification(name=dataset_name, extract_path=aeon_path)

    dataset.load()

    assert dataset.X.shape == X.shape
    assert dataset.y.shape == y.shape
    assert (X == dataset.X).all()





