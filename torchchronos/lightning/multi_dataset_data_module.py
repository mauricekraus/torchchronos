from typing import Sequence
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import lightning as L
import numpy as np
from ..datasets.prepareable_dataset import PrepareableDataset
from ..datasets.concat_dataset import ConcatDataset



class MultiDatasetDataModule(L.LightningDataModule):
    def __init__(self, datasets: Sequence[Dataset]) -> None:
        super().__init__()

        self._datasets: Sequence[Dataset]= datasets
        self.dataset: ConcatDataset = ConcatDataset(self.datasets)

    @property
    def datasets(self) -> Sequence[Dataset]:
        return self._datasets

    def prepare_data(self) -> None:
        train_data_indices: list[Sequence[int]] = []
        val_data_indices: list[Sequence[int]] = []
        test_data_indices: list[Sequence[int]] = []

        length = 0
        for dataset in self.datasets:
            if isinstance(dataset, PrepareableDataset):
                dataset.load()

            
            # train = [i for i in range(80) ]
            # val = [i for i in range(80, 90)] 
            # test = [i  for i in range(90, 100)]

            # train_data_indices.append(np.array(train) + length)	
            # val_data_indices.append(np.array(val) + length)
            # test_data_indices.append(np.array(test) + length)

            train, val, test = random_split([i for i in range(len(dataset))], [0.8, 0.1, 0.1])

            train_data_indices.append(np.array(train.indices) + length) 
            val_data_indices.append(np.array(val.indices) + length)
            test_data_indices.append(np.array(test.indices) + length)

            length += len(dataset)

        self.train_indices = np.concatenate(train_data_indices)
        self.val_indices = np.concatenate(val_data_indices)
        self.test_indices = np.concatenate(test_data_indices)

       
    def setup(self, stage:str | None =None) -> None:
        if stage == "train":
            self.train_data = Subset(self.dataset, self.train_indices)
        elif stage == "val":
            self.val_data =  Subset(self.dataset, self.val_indices)
        elif stage == "test":
            self.test_data = Subset(self.dataset, self.test_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_data, batch_size=32, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_data, batch_size=32, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_data, batch_size=32, shuffle=True)
