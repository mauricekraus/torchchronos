from collections.abc import Sequence

import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split,
)
import lightning as L

from ..datasets.concat_dataset import ConcatDataset
from ..datasets.prepareable_dataset import PrepareableDataset


class DatasetDataModule(L.LightningDataModule):
    def __init__(
        self,
        train: Dataset,
        val: Dataset | float | None = None,
        test: Dataset | float | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        super().__init__()

        self.train: Dataset = train
        self.val: Dataset | float | None = val
        self.test: Dataset | float | None = test

        self.batch_size: int = batch_size

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self._shuffle: bool = shuffle

    @property
    def datasets(self) -> Sequence[Dataset]:
        return self._datasets

    def prepare_data(self) -> None:
        for dataset in [self.train, self.val, self.test]:
            if isinstance(dataset, PrepareableDataset):
                dataset.prepare()

    def setup(self, stage: str | None = None) -> None:
        if isinstance(self.train, PrepareableDataset):
            self.train.load()
        if isinstance(self.val, PrepareableDataset):
            self.val.load()
        if isinstance(self.test, PrepareableDataset):
            self.test.load()

        if isinstance(self.val, float) and isinstance(self.test, float):
            self.train, self.val, self.test = random_split(
                self.train, [1 - self.val - self.test, self.val, self.test]
            )
        elif isinstance(self.val, float):
            self.train, self.val = random_split(self.train, [1 - self.val, self.val])
        elif isinstance(self.test, float):
            self.train, self.test = random_split(self.train, [1 - self.test, self.test])

        if stage == "fit":
            self.train_dataset = self.train
            self.val_dataset = self.val

        elif stage == "test":
            self.test_dataset = self.test

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Train dataset is not set up")

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self._shuffle,
            collate_fn=stack_collate,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError("Val dataset is not set up")

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self._shuffle,
            collate_fn=stack_collate,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Test dataset is not set up")

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self._shuffle,
            collate_fn=stack_collate,
        )


def stack_collate(batch):

    data, targets = zip(*batch)
    stacked_data = torch.cat(data)
    stacked_targets = torch.stack(targets)
    return stacked_data, stacked_targets
