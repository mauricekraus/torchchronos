from typing import Sequence
from enum import Enum, auto
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import lightning as L
import numpy as np
from ..datasets.prepareable_dataset import PrepareableDataset
from ..datasets.concat_dataset import ConcatDataset


class DatasetFrequency(Enum):
    """The relative frequency of a dataset in a collection of multiple ones."""

    ALL_EQUAL = auto()
    PROPORTIONAL_TO_SAMPLES = auto()
    ALL_TYPES_EQUAL_PROPORTIONAL_TO_SAMPLES = auto()


class ShuffleType(Enum):
    """The relative frequency of a dataset in a collection of multiple ones."""

    # Never shuffle, go sequentially through dataset list and content each
    DISABLED = auto()
    # shuffle within dataset, but go sequentially through dataset list
    WITHIN_DATASET = auto()
    # shuffle within datasets and across datasets
    ACROSS_DATASETS = auto()


class MultiDatasetDataModule(L.LightningDataModule):
    def __init__(
        self,
        train: Sequence[Dataset],
        val: Sequence[Dataset] | float,
        test: Sequence[Dataset],
        batch_size: int,
        shuffle_type: ShuffleType,
        sampling: DatasetFrequency,
    ) -> None:
        super().__init__()

        self.train: Sequence[Dataset] = train
        self.val: Sequence[Dataset] | float = val
        self.test: Sequence[Dataset] = test
        self.batch_size: int = batch_size
        self._shuffle = True if self.train_shuffle_type == ShuffleType.ACROSS_DATASETS else False

        self.train_datasets: ConcatDataset | None = None
        self.val_datasets: ConcatDataset | None = None
        self.test_datasets: ConcatDataset | None = None

    @property
    def datasets(self) -> Sequence[Dataset]:
        return self._datasets

    def prepare_data(self) -> None:
        for dataset in [*self.train, *self.val, *self.test]:
            if isinstance(dataset, PrepareableDataset):
                dataset.prepare()


    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            if isinstance(self.val, float) and self.val_datasets is not None:
                new_train: Sequence[Dataset] = []
                new_val: Sequence[Dataset] = []
                train_split = 1 - self.val_datasets

                for dataset in self.train:
                    train_split, val_split = random_split(dataset, [train_split, self.val])
                    new_train.append(train_split)
                    new_val.append(val_split)

                self.train_datasets = ConcatDataset(self._shuffle(new_train))
                self.val_datasets = ConcatDataset(self._shuffle(new_val))
            else:
                self.train_datasets = ConcatDataset(self._shuffle(new_train))
                self.val_datasets = ConcatDataset(self._shuffle(new_val))

        elif stage == "test":
            self.test_data = ConcatDataset(self._shuffle(self.test))

    def _shuffle(self, datasets: Sequence[Dataset], shuffle_type: ShuffleType) -> Sequence[Dataset]:
        if shuffle_type == ShuffleType.DISABLED:
            return datasets
        elif shuffle_type == ShuffleType.WITHIN_DATASET:
            return [Subset(dataset, list(np.random.permutation(len(dataset)))) for dataset in datasets]
        if self.self.train_shuffle_type == ShuffleType.ACROSS_DATASETS:
            return datasets

    def train_dataloader(self) -> DataLoader:
        
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
