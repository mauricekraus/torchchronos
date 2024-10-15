from torchchronos.datasets.aeon_datasets import AeonClassificationDataset


def test_aeon_classification_dataset():
    AeonClassificationDataset(name="GunPoint")
