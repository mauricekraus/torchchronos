import math
from pathlib import Path
from typing import Literal
from lightning import LightningDataModule

from ..typing import DatasetSplit

from ..datasets import UCRUEADataset
from ..download import download_uea_ucr
from torch.utils.data import random_split, DataLoader

from ..utils import swap_batch_seq_collate_fn

from ..transforms import Transform


class UCRUEADataModule(LightningDataModule):
    """UCR/UEA dataset datamodule

    Args:
        name : str
            Name of the dataset.
        split_ratio : float | tuple[float, float], optional (default: (0.75, 0.15))
            Ratio of train, val, test split of the entire dataset.
            If a tuple of two floats is passed, we define the ratio of train and validation splits on the entire dataset i.e.
            TRAIN_SPLIT + TEST_SPLIT of the original UEA_UCR dataset. For instance,
            if the TRAIN_SPLIT has a length of 10 and TEST_SPLIT has a length of 2, and we choose (0.75, 0.15),
            this will result in 75% train, 15% validation, 10% test, and thus 9 train, 2 val, 1 test.
            If a float is passed, we define the ratio of train and validation splits on the TRAIN_SPLIT of the original UEA_UCR dataset, and the TEST_SPLIT will remain untouched.
            For instance, if the TRAIN_SPLIT has a length of 10 and TEST_SPLIT has a length of 2, and we choose 0.75,
            this will result in 75% train, 25% validation, 10% test, and thus 7 train, 3 val, 2 test.
        batch_size : int, optional (default: 32)
            The batch size to use for the dataloaders.
        transform : Transform or None, optional (default: None)
            The transform to apply to the dataset.

    Attributes:
        cache_dir : pathlib.Path
            Directory where the data will be downloaded and stored.
    """

    def __init__(
        self,
        name: str,
        split_ratio: tuple[float, float] | float = (0.75, 0.15),
        batch_size: int = 32,
        transform: Transform | None = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.cache_dir = Path(".cache") / "data"
        self.split_ratio = split_ratio
        self.name = name
        self.transform = transform

    def prepare_data(self):
        download_uea_ucr(self.name, self.cache_dir)

    def label_from_index(self, index: int) -> str:
        assert self.__label_from_index is not None, "You need to call setup first"
        return self.__label_from_index(index)

    @property
    def num_classes(self) -> int:
        assert self.__num_classes is not None, "You need to call setup first"
        return self.__num_classes

    @property
    def dimensions(self) -> int:
        assert self.__dimensions is not None, "You need to call setup first"
        return self.__dimensions

    @property
    def series_length(self) -> int:
        assert self.__series_length is not None, "You need to call setup first"
        return self.__series_length

    @property
    def equal_length(self) -> bool:
        assert self.__equal_length is not None, "You need to call setup first"
        return self.__equal_length

    @property
    def univariate(self) -> bool:
        assert self.__univariate is not None, "You need to call setup first"
        return self.__univariate

    @property
    def missing(self) -> bool:
        assert self.__missing is not None, "You need to call setup first"
        return self.__missing

    def setup(self, stage=None):
        if isinstance(self.split_ratio, float):
            train_split_dataset = UCRUEADataset(
                self.name,
                self.cache_dir,
                split=DatasetSplit.TRAIN,
                transform=self.transform,
            )
            val_size = round(1 - self.split_ratio, 2)
            self.train_dataset, self.val_dataset = random_split(
                train_split_dataset, [self.split_ratio, val_size]
            )
            self.test_dataset = UCRUEADataset(
                self.name,
                self.cache_dir,
                split=DatasetSplit.TEST,
                transform=self.transform,
            )

            self.__label_from_index = train_split_dataset.label_from_index
            self.__num_classes = train_split_dataset.num_classes
            self.__dimensions = train_split_dataset.dimensions
            self.__series_length = train_split_dataset.series_length
            self.__equal_length = train_split_dataset.equal_length
            self.__univariate = train_split_dataset.univariate
            self.__missing = train_split_dataset.missing
        else:
            dataset = UCRUEADataset(
                ds_name=self.name, path=self.cache_dir, transform=self.transform
            )
            self.__label_from_index = dataset.label_from_index
            self.__num_classes = dataset.num_classes
            self.__dimensions = dataset.dimensions
            self.__series_length = dataset.series_length
            self.__equal_length = dataset.equal_length
            self.__univariate = dataset.univariate
            self.__missing = dataset.missing

            # split dataset
            test_size = round(1 - self.split_ratio[0] - self.split_ratio[1], 2)
            if math.isclose(test_size, 0.0):
                self.train_dataset, self.val_dataset = random_split(
                    dataset, [self.split_ratio[0], self.split_ratio[1]]
                )
            else:
                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    dataset, [self.split_ratio[0], self.split_ratio[1], test_size]
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=swap_batch_seq_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=swap_batch_seq_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=swap_batch_seq_collate_fn,
        )
