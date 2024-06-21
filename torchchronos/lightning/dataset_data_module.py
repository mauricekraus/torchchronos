"""Module for the DatasetDataModule class."""

import lightning as L
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split,
)

from torchchronos.datasets import PrepareableDataset


class DatasetDataModule(L.LightningDataModule):
    """A data module for for multiple datasets.

    With this data module, you can train your model on one dataset and test or validate on another dataset.
    This can also be done in combindation with the Concat dataset, where multiple datasets can be combined
    into one.

    Args:
        train: The training dataset.
        val: The validation dataset. This can either be a float or a dataset. If it is a float, it will be
            used as the fraction of the training dataset to be used for validation.
        test: The test dataset. This can either be a float or a dataset. If it is a float, it will be used
            as the fraction of the training dataset to be used for testing.
        batch_size: The batch size.
        shuffle: Whether to shuffle the data.

    Examples:
    A simple example where a single dataset is used for training and a fraction is passed for the val and
    test splits.

    >>> from torchchronos.lightning import DatasetDataModule
    >>> from torchchronos.datasets import AeonClassificationDataset
    >>>
    >>> gun_point = AeonClassificationDataset("GunPoint")
    >>> ddm = DatasetDataModule(gun_point, val=0.2, test=0.2)
    >>> ddm.prepare_data()
    >>> ddm.setup("fit")
    >>> train_loader = ddm.train_dataloader()

    An example where we have two datasets, one for training and one for testing. Those do not have to be
    the same dataset. It is not enforced that the datasets have the same size, but it makes sence for
    training a model to pad them to the same size. This however have to be done in the dataset itself, as the
    dataloader will only make the handling of the data and setup of the correct dataset splits.

    >>> gun_point = AeonClassificationDataset("GunPoint")
    >>> wafer = AeonClassificationDataset("Wafer")
    >>> ddm = DatasetDataModule(gun_point, test=wafer)
    >>> ddm.prepare_data()
    >>> ddm.setup("fit")
    >>> ddm.setup("test")
    >>>
    >>> train_loader = ddm.train_dataloader()
    >>>
    >>> for batch in train_loader:
    ...     data, targets = batch
    >>>
    >>> print(len(ddm.train_dataset))
    200
    >>> print(len(ddm.test_dataset))
    7164
    """

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

        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None
        self._test_dataset: Dataset | None = None
        self._shuffle: bool = shuffle

    @property
    def train_dataset(self) -> Dataset:
        """The training dataset."""
        if self._train_dataset is None:
            raise ValueError("Train dataset is not set up")
        return self._train_dataset

    @property
    def val_dataset(self) -> Dataset:
        """The validation dataset."""
        if self._val_dataset is None:
            raise ValueError("Validation dataset is not set up")
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        """The test dataset."""
        if self._test_dataset is None:
            raise ValueError("Test dataset is not set up")
        return self._test_dataset

    def prepare_data(self) -> None:
        """Prepare the datasets for usage.

        Since the main feature of this library are PrepareableDatasets, this method will call the prepare
        method on the datasets.
        """
        for dataset in [self.train, self.val, self.test]:
            if isinstance(dataset, PrepareableDataset):
                dataset.prepare()

    def setup(self, stage: str | None = None) -> None:
        """Set up the datasets for usage.

        First all load methods are called on the datasets, then the datasets are split into train, val and
        test sets.
        """
        for dataset in [self.train, self.val, self.test]:
            if isinstance(dataset, PrepareableDataset):
                dataset.load()

        if isinstance(self.val, float) and isinstance(self.test, float):
            self.train, self.val, self.test = random_split(
                self.train, [1 - self.val - self.test, self.val, self.test]
            )
        elif isinstance(self.val, float):
            self.train, self.val = random_split(self.train, [1 - self.val, self.val])
        elif isinstance(self.test, float):
            self.train, self.test = random_split(self.train, [1 - self.test, self.test])

        if stage == "fit":
            self._train_dataset = self.train
            if isinstance(self.val, Dataset):
                self._val_dataset = self.val

        elif stage == "test":
            if isinstance(self.test, Dataset):
                self._test_dataset = self.test

    def train_dataloader(self) -> DataLoader:
        """Get the train dataloader."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not set up")

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self._shuffle,
            collate_fn=stack_collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not set up, or does not exist.")

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self._shuffle,
            collate_fn=stack_collate,
        )

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader."""
        if self.test_dataset is None:
            raise ValueError("Test dataset is not set up, or does not exist.")

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self._shuffle,
            collate_fn=stack_collate,
        )


def stack_collate(batch):
    """Collate function for stacking data and targets."""
    data, targets = zip(*batch)
    stacked_data = torch.cat(data)
    stacked_targets = torch.stack(targets)
    return stacked_data, stacked_targets
