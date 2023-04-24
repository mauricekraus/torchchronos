from torchchronos.typing import DatasetSplit
from torchchronos.datasets import UCRUEADataset
from torchchronos.download import download_uea_ucr


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
