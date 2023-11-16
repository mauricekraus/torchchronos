import warnings
from collections.abc import Generator, Sequence, Sized
from enum import Enum, auto
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from hydra_zen import MISSING, hydrated_dataclass
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    Dataset,
    Sampler,
    SequentialSampler,
    WeightedRandomSampler,
    random_split,
)
from torchchronos.datasets import UCRUEADataset
from torchchronos.download import download_uea_ucr
from torchchronos.datasets.tfc_pretrain import download_tfc_pretrain, TFCPretrainDataset
from torchchronos.typing import DatasetSplit



# from ..util.decorators import DATA_PREFIX_REGISTRY
# from ..util.load_datasets import get_fixed_length_datasets
# from ..util.permutation_list import make_ds_augmentation_permutation_list
# from .augmentations import (
#     Augmentation,
#     AugmentationConfig,
#     make_tstcc_augmentation_config,
#     make_xd_mixup_augmentation_config,
#     make_tfc_augmentation_config,
# )
# from .torchchronos_augmenations import (
#     TransformConfig,
#     StandardScalerConfig,
#     NormalScalerConfig,
# )
# from .augmented_dataset import AugmentedDataset
# from .base_data_module import (
#     BaseDatasetEntryConfig,
#     MultiBaseDataModule,
#     MultiBaseDataModuleConfig,
# )
# from .compound_datasets import ConcatDataset, ShuffledDataset
# from .ts_tcc_dataset import TSTCCDataset


class MultiDatasetDataModule(MultiBaseDataModule):
    """DataModule for multiple datasets.

    Warning:
        The notion of "eopchs" is usually defined as a single pass over the training
        set. However, in this case, we have multiple datasets, so we need to define
        what a single epoch means. We can either define it as a single pass over all
        datasets (in which case we have to sample from each dataset with the same
        frequency), or we can define it as a single pass over the training set of
        each dataset (in which case we have to sample from each dataset with a
        frequency proportional to the number of samples in the training set).
        Both are supported by this class, see ``dataset_frequency``.
        In any case, the validation and test sets are always sampled per-instance.

    Warning:
        When setting ``dataset_frequency=ALL_EQUAL``, some samples might be drawn
        more than once within a single epoch. This is because if the number of
        samples in the training set of a dataset is musch less than the batch size,
        we have to sample with replacement in otder to maintain dataset ratios.
        "Epoch" has little to no meaning in this case, so if it is important,
        one should specify a sufficiently small batch size.

    Note:
        Due to the different sampling strategies, one should usually specify a
        number of update steps instead of epochs when training with this class.
        An epoch can be very short, e.g. if even just a single dataset is small.

    Shapes:
        The returned data is in the form ``(batch_size, sequence_len, num_features)``.

    The test & val sets are always sampled per-instance from the test sets of the
    individual datasets (like the `PROPORTIONAL_TO_SAMPLES` mode).
    """

    # Overwrite the type hints
    dataset_entries: list["SingleDatasetEntryConfig"]

    length: int  # The length of the training set (all datasets combined)
    weights: Sequence[float] | None  # One weight per entry per dataset

    def __init__(self, **kwargs) -> None:
        params = MultiDatasetDataModuleConfig(**kwargs)
        self.dataset_frequency = params.dataset_frequency
        self.ucruea_split_ratio = (
            params.split_ratio
        )  # TODO this should apply to all datasets
        self.show_download_progress_bar = params.show_download_progress_bar
        self.shuffle_type = params.shuffle_type
        super().__init__(
            batch_size=params.batch_size,
            data_path=params.data_path,
            drop_last=params.drop_last,
            num_workers=params.num_workers,
            dataset_entries=cast(list[BaseDatasetEntryConfig], params.dataset_entries),
            num_classes=params.num_classes,
            sequence_len=params.sequence_len,
        )

    def prepare_data(self) -> None:
        with Progress(
            SpinnerColumn(),  # Show that the process did not freeze
            TextColumn("[progress.description]{task.description}"),  # default
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            disable=not self.show_download_progress_bar,
        ) as progress:
            for entry in progress.track(
                self.dataset_entries,
                description="Downloading datasets & warming up cache",
            ):
                print(f"loading dataset: {entry.name} …")
                match entry.source:
                    case DatasetRepository.UCR_UEA:
                        download_uea_ucr(entry.name, Path(entry.cache_dir))
                        # We have to instantiate the datasets here, in order to warm up the cache
                        with warnings.catch_warnings():
                            warnings.simplefilter(
                                "ignore", category=pd.errors.PerformanceWarning
                            )
                            UCRUEADataset(
                                entry.name,
                                Path(entry.cache_dir),
                                split=DatasetSplit.TRAIN,
                            )
                            UCRUEADataset(
                                entry.name,
                                Path(entry.cache_dir),
                                split=DatasetSplit.TEST,
                            )
                    case DatasetRepository.TS_TCC:
                        # TODO: we could check that the data is already there
                        pass  # Nothing to do here
                    case DatasetRepository.TORCHCHRONOS_TFC:
                        download_tfc_pretrain(entry.cache_dir, entry.name)
                    case _:
                        raise NotImplementedError(
                            f"Unknown dataset source: {entry.source}"
                        )

    def setup(self, stage: str) -> None:
        train_datasets = []
        val_datasets = []
        test_datasets = []
        fractions = []

        for entry in self.dataset_entries:
            augmentation = cast(Augmentation, entry.augmentation)

            def wrapped(dataset: Dataset) -> Dataset:
                reordered = (
                    ShuffledDataset(dataset)
                    if self.shuffle_type == ShuffleType.WITHIN_DATASET
                    else dataset
                )
                return AugmentedDataset(reordered, augmentation)

            match entry.source:
                case DatasetRepository.UCR_UEA:
                    match stage:
                        case "fit":
                            train_split_dataset = UCRUEADataset(
                                entry.name,
                                Path(entry.cache_dir),
                                split=DatasetSplit.TRAIN,
                                transform=entry.transform,
                            )
                            val_size = round(1 - self.ucruea_split_ratio, 2)
                            train_dataset, val_dataset = random_split(
                                train_split_dataset, [self.ucruea_split_ratio, val_size]
                            )

                            train_datasets.append(wrapped(train_dataset))
                            if "Gesture" in entry.name:
                                # TODO workaround:
                                test_dataset = UCRUEADataset(
                                    entry.name,
                                    Path(entry.cache_dir),
                                    split=DatasetSplit.TEST,
                                    transform=entry.transform,
                                )
                                val_datasets.append(wrapped(test_dataset))
                            else:
                                val_datasets.append(wrapped(val_dataset))

                        case "test":
                            test_dataset = UCRUEADataset(
                                entry.name,
                                Path(entry.cache_dir),
                                split=DatasetSplit.TEST,
                                transform=entry.transform,
                            )
                            test_datasets.append(wrapped(test_dataset))

                        case _:
                            raise NotImplementedError(f"Unknown stage: {stage}")

                case DatasetRepository.TS_TCC:
                    # TODO: move this trick into the TSTCCDataset class
                    if entry.name in {"FD-A", "FD-B", "FD-C", "FD-D"}:
                        ds_path = self.data_path / entry.name.replace("-", "/")
                    else:
                        ds_path = self.data_path / entry.name
                    match stage:
                        case "fit":
                            # We do not pass the fraction here, and instead do it the same way as
                            # for the UCR-UEA datasets for uniformity
                            # TODO: we need to apply the transform too
                            train_datasets.append(
                                wrapped(TSTCCDataset(ds_path / "train.pt"))
                            )
                            val_datasets.append(
                                wrapped(TSTCCDataset(ds_path / "val.pt"))
                            )
                        case "test":
                            test_datasets.append(
                                wrapped(TSTCCDataset(ds_path / "test.pt"))
                            )
                        case _:
                            raise NotImplementedError(f"Unknown stage: {stage}")

                case DatasetRepository.TORCHCHRONOS_TFC:
                    match stage:
                        case "fit":
                            train_datasets.append(
                                wrapped(
                                    TFCPretrainDataset(
                                        entry.name,
                                        entry.cache_dir,
                                        DatasetSplit.TRAIN,
                                        entry.transform,
                                    )
                                )
                            )
                            val_datasets.append(
                                wrapped(
                                    TFCPretrainDataset(
                                        entry.name,
                                        entry.cache_dir,
                                        DatasetSplit.VAL,
                                        entry.transform,
                                    )
                                )
                            )
                        case "test":
                            test_datasets.append(
                                wrapped(
                                    TFCPretrainDataset(
                                        entry.name,
                                        entry.cache_dir,
                                        DatasetSplit.TEST,
                                        entry.transform,
                                    )
                                )
                            )
                case _:
                    raise NotImplementedError(f"Unknown dataset source: {entry.source}")

            fractions.append(entry.fraction)

        if stage == "fit":
            self.train_dataset = ConcatDataset(train_datasets, fractions)
            self.val_dataset = ConcatDataset(val_datasets)

            lengths = np.array(self.train_dataset.lengths, dtype=np.int64)
            assert (lengths > 0).all(), "All lengths should be strictly positive"

            # Note: the weights do not need to be normalized
            match self.dataset_frequency:
                case DatasetFrequency.ALL_EQUAL:
                    self.length = lengths.min().item()
                    weights = lengths.sum().astype(np.float32) / lengths
                    self.weights = np.repeat(weights, lengths)  # type: ignore
                case DatasetFrequency.PROPORTIONAL_TO_SAMPLES:
                    self.length = self.train_dataset.total_length
                    self.weights = None
                case DatasetFrequency.ALL_TYPES_EQUAL_PROPORTIONAL_TO_SAMPLES:
                    self.length = self.train_dataset.total_length
                    all_names = [
                        entry.name for entry in self.dataset_entries
                    ]  # The order matters
                    df = get_fixed_length_datasets_with_others()
                    df = df[df["Dataset"].isin(all_names)]
                    df = df.set_index("Dataset").loc[all_names].reset_index()
                    df["Train_size"] = df["Train_size"] * fractions
                    weight_per_dataset = 1 / df.groupby("Type")["Train_size"].transform(
                        lambda x: x.sum()
                    )
                    self.weights = np.repeat(weight_per_dataset.to_numpy(), lengths)  # type: ignore
                case _:
                    raise NotImplementedError(
                        f"Unknown dataset frequency: {self.dataset_frequency}"
                    )

        elif stage == "test":
            self.test_dataset = ConcatDataset(test_datasets)

        else:
            raise ValueError(f"Unknown stage: {stage}")

        self._shuffle_in_dataloader = self.shuffle_type == ShuffleType.ACROSS_DATASETS

    def _get_individual_sampler(self, dataset: Sized, shuffle: bool) -> Sampler:
        # We try to mimic the Pytorch default behavior here
        if shuffle and self._shuffle_in_dataloader:
            if self.weights is None:
                return RandomSampler(dataset, replacement=False)
            else:
                if not len(self.weights) == len(dataset):
                    raise ValueError(
                        "The length of weights should be equal to the length of dataset."
                    )
                return WeightedRandomSampler(
                    self.weights,
                    num_samples=len(dataset),
                    replacement=True,  # See class documentation
                )
        else:
            return SequentialSampler(dataset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            sampler=self._get_individual_sampler(self.train_dataset, shuffle=True),
            num_workers=self.num_workers,
            multiprocessing_context=self.multiprocessing_context,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            sampler=self._get_individual_sampler(self.val_dataset, shuffle=False),
            num_workers=self.num_workers,
            multiprocessing_context=self.multiprocessing_context,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            sampler=self._get_individual_sampler(self.test_dataset, shuffle=False),
            num_workers=self.num_workers,
            multiprocessing_context=self.multiprocessing_context,
        )


