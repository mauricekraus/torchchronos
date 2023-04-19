from pathlib import Path
from lightning import LightningDataModule

from torch.utils.data import DataLoader

from ..utils import swap_batch_seq_collate_fn

from ..transforms import Transform
from ..typing import DatasetSplit
from ..datasets import TFCPretrainDataset


class TFCPretrainDataModule(LightningDataModule):
    """See :class:`~torchchronos.datasets.TFCPretrainDataset` for more details."""

    def __init__(
        self,
        name: str,
        batch_size: int = 32,
        transform: Transform | None = None,
    ):
        super().__init__()

        self.name = name
        self.batch_size = batch_size
        self.transform = transform

        self.cache_dir = Path(".cache") / "data"

    def prepare_data(self) -> None:
        # The test split does not matter
        TFCPretrainDataset(
            self.name, self.cache_dir, DatasetSplit.TRAIN, transform=None
        )

    def setup(self, stage: str | None = None) -> None:
        match stage:
            case "fit":
                self.train_dataset = TFCPretrainDataset(
                    self.name, self.cache_dir, DatasetSplit.TRAIN, self.transform
                )
                self.val_dataset = TFCPretrainDataset(
                    self.name, self.cache_dir, DatasetSplit.VAL, self.transform
                )
            case "validate":
                self.val_dataset = TFCPretrainDataset(
                    self.name, self.cache_dir, DatasetSplit.VAL, self.transform
                )
            case "test":
                self.test_dataset = TFCPretrainDataset(
                    self.name, self.cache_dir, "test", self.transform
                )
            case _:
                # Handle any other cases or raise an error if needed
                raise ValueError(f"Invalid stage: {stage}")

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
