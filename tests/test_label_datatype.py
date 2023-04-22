from pathlib import Path
from lightning import seed_everything
import pytest
import numpy as np
from torchchronos.download import download_uea_ucr

from torchchronos.datasets import UCRUEADataset
from tests.data_class import Dataset


@pytest.fixture
def dataset_fixture():
    return [
        Dataset("GunPoint", 50, 150, 2),
        Dataset("ACSF1", 100, 100, 10),
        Dataset("NATOPS", 180, 180, 6),
        Dataset("PhonemeSpectra", 3315, 3353, 39),
        Dataset("ShapesAll", 600, 600, 60),
        Dataset("DistalPhalanxOutlineCorrect", 600, 276, 2),
    ]


def test_label_datatype_factorization(dataset_fixture: list[Dataset]):
    seed_everything(12)

    for dataset in dataset_fixture:  # TODO change
        cache_dir = Path(".cache") / "data"
        download_uea_ucr(cache_dir, dataset.name)

        ds = UCRUEADataset(dataset.name, path=cache_dir)

        assert len(np.unique(ds.y_labels)) == dataset.num_classes
        assert len(np.unique(ds.ys)) == dataset.num_classes
        assert ds.num_classes == dataset.num_classes


# if __name__ == "__main__":
# test_label_datatype_factorization(dataset_fixture())
