from collections.abc import Sequence
from enum import Enum, auto

import numpy as np
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
    Subset,
    WeightedRandomSampler,
    random_split,
)

import lightning as L

from ..datasets.concat_dataset import ConcatDataset
from ..datasets.prepareable_dataset import PrepareableDataset


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
        self.shuffle_type: ShuffleType = shuffle_type
        self._shuffle = True if shuffle_type == ShuffleType.ACROSS_DATASETS else False
        self.sampling: DatasetFrequency = sampling

        self.train_dataset: ConcatDataset | None = None
        self.train_weights: Sequence[float] | None = None

        self.val_dataset: ConcatDataset | None = None
        self.val_weights: Sequence[float] | None = None

        self.test_dataset: ConcatDataset | None = None
        self.test_weights: Sequence[float] | None = None

    @property
    def datasets(self) -> Sequence[Dataset]:
        return self._datasets

    def prepare_data(self) -> None:
        val = [] if isinstance(self.val, float) else self.val
        for dataset in self.train + val + self.test:
            try:
                dataset.domain
            except AttributeError as e:
                raise AttributeError(
                    f"Dataset {dataset} has no attribute 'domain'. " "Please add the attribute to the class"
                ) from e
        for dataset in self.train + val + self.test:
            if isinstance(dataset, PrepareableDataset):
                dataset.prepare()
                dataset.load()

        # Find the maximum length of all datasets
        max_len = -1
        for dataset in self.train + val + self.test:
            sample = dataset[0]
            if isinstance(sample, tuple):
                data, target = sample
                max_len = max(max_len, data.shape[-1])

        # Pad all datasets to the maximum length
        for dataset in self.train + val + self.test:
            pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            if isinstance(self.val, float):
                new_train: Sequence[Dataset] = []
                new_val: Sequence[Dataset] = []
                train_split_frequency = 1 - self.val

                for dataset in self.train:
                    train_split, val_split = random_split(
                        dataset, [float(train_split_frequency), float(self.val)]
                    )
                    new_train.append(train_split)
                    new_val.append(val_split)

                self.train = new_train
                self.val = new_val

                self.train_dataset = ConcatDataset(self._shuffle_data(new_train, self.shuffle_type))
                self.val_dataset = ConcatDataset(self._shuffle_data(new_val, self.shuffle_type))
            else:
                self.train_dataset = ConcatDataset(self._shuffle_data(self.train, self.shuffle_type))
                self.val_dataset = ConcatDataset(self._shuffle_data(self.val, self.shuffle_type))

            self.train_weights = self._get_frequency_weights(self.train, len(self.train_dataset))
            self.val_weights = self._get_frequency_weights(self.val, len(self.val_dataset))

        elif stage == "test":
            self.test_dataset = ConcatDataset(self.test, self.shuffle_type)
            self.test_weights = self._get_frequency_weights(self.test, len(self.test_dataset))

    def _shuffle_data(self, datasets: Sequence[Dataset], shuffle_type: ShuffleType) -> Sequence[Dataset]:
        if shuffle_type == ShuffleType.DISABLED:
            return datasets
        elif shuffle_type == ShuffleType.WITHIN_DATASET:
            return [Subset(dataset, list(np.random.permutation(len(dataset)))) for dataset in datasets]
        if self.self.train_shuffle_type == ShuffleType.ACROSS_DATASETS:
            return datasets

    def _get_frequency_weights(self, datasets: Sequence[Dataset], dataset_length) -> Sequence[float]:
        weights: Sequence[float] = []
        if self.sampling == DatasetFrequency.ALL_EQUAL:
            weights = np.repeat(1 / dataset_length, dataset_length)
        elif self.sampling == DatasetFrequency.PROPORTIONAL_TO_SAMPLES:
            for dataset in datasets:
                weights.append(np.repeat(1 / len(dataset), len(dataset)))
            weights = np.concatenate(weights)
        elif self.sampling == DatasetFrequency.ALL_TYPES_EQUAL_PROPORTIONAL_TO_SAMPLES:
            # Todo: implement later
            raise NotImplementedError

        return weights

    def _get_sampler(self, weights: Sequence[float], dataset) -> Sampler:
        # prüfen auf self.frequency und dann ggf. RandomSampler zurückgeben
        return WeightedRandomSampler(weights, len(dataset))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            sampler=self._get_sampler(self.train_weights, self.train_dataset),
            shuffle=self._shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            sampler=self._get_sampler(self.val_weights, self.val_dataset),
            shuffle=self._shuffle,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            sampler=self._get_sampler(self.test_weights, self.test_dataset),
            shuffle=self._shuffle,
        )
