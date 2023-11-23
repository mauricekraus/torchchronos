import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule

from ..datasets.prepareable_dataset import PreparedDataset
from ..datasets.multidatasetdataset import (
    MultiDatasetDataet,
    DatasetFrequency,
    ShuffleType
)


class MultiDatasetDataModule(LightningDataModule):

    def __init__(self, datasets: list[Dataset]) -> None:
        super().__init__()

        self.dataset = datasets

    def prepare_data(self) -> None:
        self.train_data = []
        self.val_data = []
        self.test_data = []

        for dataset in self.dataset:
            if isinstance(dataset, PreparedDataset):
                dataset.load()

            train, val, test = random_split(
                dataset,
                [0.8, 0.1, 0.1])

            self.train_data.append(train)
            self.val_data.append(val)
            self.test_data.append(test)

        # check if present
        # if not create the datasets
        train_data_set = MultiDatasetDataet(
            self.train_data,
            DatasetFrequency.ALL_EQUAL,
            ShuffleType.DISABLED
        )
        val_data_set = MultiDatasetDataet(
            self.val_data,
            DatasetFrequency.ALL_EQUAL,
            ShuffleType.DISABLED
        )
        test_data_set = MultiDatasetDataet(
            self.test_data,
            DatasetFrequency.ALL_EQUAL,
            ShuffleType.DISABLED
        )

        torch.save(train_data_set, "data/multi_datasets/train_data_set.pt")
        torch.save(val_data_set, "data/multi_datasets/val_data_set.pt")
        torch.save(test_data_set, "data/multi_datasets/test_data_set.pt")

    def setup(self, stage=None) -> None:
        if stage == "train":
            return torch.load("data/multi_datasets/train_data_set.pt")
        elif stage == "val":
            return torch.load("data/multi_datasets/val_data_set.pt")
        elif stage == "test":
            return torch.load("data/multi_datasets/test_data_set.pt")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_data, batch_size=32, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_data, batch_size=32, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_data, batch_size=32, shuffle=True)
