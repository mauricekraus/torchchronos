import math
from pathlib import Path
from typing import Callable, Optional, Union
from lightning import LightningDataModule
import torch

from torchchronos.datasets.ucr_uea_dataset import UCRUEADataset
from torchchronos.download import _download_uea_ucr
from torch.utils.data import random_split, DataLoader

from ..utils import swap_batch_seq_collate_fn

from ..transforms import Compose


class UCRUEAModule(LightningDataModule):
    def __init__(
        self,
        name: str,
        split_ratio: tuple[float, float] = (0.75, 0.15),
        batch_size: int = 32,
        transform: Optional[
            Union[Callable[[torch.Tensor], torch.Tensor], Compose]
        ] = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.cache_dir = Path(".cache/data")
        self.split_ratio = split_ratio
        self.name = name
        self.transform = transform

    def prepare_data(self):
        _download_uea_ucr(self.cache_dir, self.name)

    def label_from_float_index(self, index: float) -> str:
        assert self.__label_from_float_index is not None, "You need to call setup first"
        return self.__label_from_float_index(index)

    def setup(self, stage=None):
        dataset = UCRUEADataset(self.name, self.cache_dir, self.transform)
        self.__label_from_float_index = dataset.label_from_float_index

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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=swap_batch_seq_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=swap_batch_seq_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=swap_batch_seq_collate_fn,
        )
