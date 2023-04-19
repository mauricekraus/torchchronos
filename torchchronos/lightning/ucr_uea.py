import math
from pathlib import Path
from lightning import LightningDataModule

from ..datasets import UCRUEADataset
from ..download import download_uea_ucr
from torch.utils.data import random_split, DataLoader

from ..utils import swap_batch_seq_collate_fn

from ..transforms import Transform


class UCRUEADataModule(LightningDataModule):
    def __init__(
        self,
        name: str,
        split_ratio: tuple[float, float] = (0.75, 0.15),
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
        download_uea_ucr(self.cache_dir, self.name)

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

    def setup(self, stage=None):
        dataset = UCRUEADataset(self.name, self.cache_dir, self.transform)
        self.__label_from_index = dataset.label_from_index
        self.__num_classes = dataset.num_classes
        self.__dimensions = dataset.dimensions
        self.__series_length = dataset.series_length
        self.__equal_length = dataset.equal_length
        self.__univariate = dataset.univariate

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
