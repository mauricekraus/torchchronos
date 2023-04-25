import pytest
from torchchronos.errors import MissingValueError
from torchchronos.typing import DatasetSplit
from torchchronos.datasets import UCRUEADataset
from torchchronos.download import download_uea_ucr
import torch


def test_UCR_dataset_loading(tmp_path) -> None:
    """Smoke test for the different variants."""

    # Prepare the dataset
    name = "AtrialFibrillation"
    download_uea_ucr(name, tmp_path)

    # With torchchronos cache (but is is cold)
    dataset = UCRUEADataset(
        ds_name=name, path=tmp_path, split=DatasetSplit.TEST, torchchronos_cache=True
    )
    assert len(dataset) == 15

    # Without torchchronos cache
    dataset = UCRUEADataset(
        ds_name=name, path=tmp_path, split=DatasetSplit.TEST, torchchronos_cache=False
    )
    assert len(dataset) == 15

    # With torchchronos cache (and is is hot)
    dataset = UCRUEADataset(
        ds_name=name, path=tmp_path, split=DatasetSplit.TEST, torchchronos_cache=True
    )
    assert len(dataset) == 15


def test_UCR_dataset_loading_strings(tmp_path) -> None:
    """Smoke test for datasets that contain strings as labels."""

    # Prepare the dataset
    name = "Handwriting"
    download_uea_ucr(name, tmp_path)

    # Train is a bit smaller than test
    dataset = UCRUEADataset(ds_name=name, path=tmp_path, split=DatasetSplit.TRAIN)
    assert len(dataset) == 150, len(dataset)
    x, y = dataset[0]
    assert x.shape == (152, 3), x.shape
    assert y.shape == (1,) or y.shape == (), y.shape
    assert y.dtype == torch.long, y.dtype


def test_UCR_dataset_missing_raise(tmp_path) -> None:
    # Prepare missing dataset
    name = "DodgerLoopDay"
    download_uea_ucr(name, tmp_path)

    # Test that it raises
    with pytest.raises(
        MissingValueError,
        match=f"Dataset {name} contains NaN values. If this is intended behavior, set `raise_on_missing=False`",
    ):
        _ = UCRUEADataset(
            ds_name=name, path=tmp_path, split=DatasetSplit.TRAIN, raise_on_missing=True
        )

    dataset = UCRUEADataset(
        ds_name=name,
        path=tmp_path,
        split=DatasetSplit.TRAIN,
        raise_on_missing=False,
    )

    assert dataset.missing
    assert torch.count_nonzero(torch.isnan(dataset.xs)) > 0
