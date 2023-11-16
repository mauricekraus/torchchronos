from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from ..datasets.download_datasets import load_dataset
from ..datasets.multidatasetdataset import MultiDatasetDataet, DatasetFrequency, ShuffleType


class MultiDatasetDataModule(LightningDataModule):
    
    def __init__(self, dataset_names: list[str]) -> None:
        super().__init__()

        self.dataset_names = dataset_names


    def prepare_data(self) -> None:
        for dataset_name in self.dataset_names:
            load_dataset(dataset_name, split=None)
        
    def setup(self, stage=None) -> None:
        self.train_data = []
        self.val_data = []
        self.test_data = []

        for dataset_name in self.dataset_names:
            train, val, test = random_split(load_dataset(dataset_name, split=None), [0.8, 0.1, 0.1])
            self.train_data.append(train)
            self.val_data.append(val)
            self.test_data.append(test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_data, batch_size=32, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_data, batch_size=32, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_data, batch_size=32, shuffle=True)
