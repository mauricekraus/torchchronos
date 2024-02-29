import tempfile
import pytest
from pathlib import Path
import torch

from torchchronos.datasets.cached_datasets import CachedDataset
from torchchronos.datasets.aeon_datasets import AeonClassificationDataset
from torchchronos.datasets.utils import save_dataset


def test_prepare():
    with tempfile.TemporaryDirectory() as tmpdirname:
        dataset = AeonClassificationDataset(name="GunPoint")
        dataset.prepare()
        dataset.load()
        save_dataset(dataset, "test_dataset", Path(tmpdirname))

        dataset = CachedDataset(name="wrong_file", path=tmpdirname)
        with pytest.raises(FileNotFoundError):
            dataset.prepare()

        dataset = CachedDataset(name="test_dataset", path=tmpdirname)
        assert not dataset.is_prepared
        dataset.prepare()
        assert dataset.is_prepared


def test_load():
    with tempfile.TemporaryDirectory() as tmpdirname:
        dataset = AeonClassificationDataset(name="GunPoint")
        dataset.prepare()
        dataset.load()
        save_dataset(dataset, "test_dataset", Path(tmpdirname))

        cached_dataset = CachedDataset(name="test_dataset", path=tmpdirname)
        cached_dataset.prepare()
        assert not cached_dataset.is_loaded
        cached_dataset.load()
        assert cached_dataset.is_loaded
        assert len(dataset) == len(cached_dataset)
        print(dataset[:][0].shape)
        print(cached_dataset[:][0].shape)

        assert torch.all(torch.eq(dataset[:][0], cached_dataset[:][0]))
        assert torch.all(torch.eq(dataset[:][1], cached_dataset[:][1]))
