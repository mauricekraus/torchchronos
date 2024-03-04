from pathlib import Path
from typing import Optional

import torch
import numpy as np
from aeon.datasets._data_loaders import load_classification

from ..transforms import Compose, Transform, Identity, ToTorchTensor, LabelTransform
from .prepareable_dataset import PrepareableDataset


class AeonClassificationDataset(PrepareableDataset):
    def __init__(
        self,
        name: str,
        split: Optional[str] = None,
        path: Path | str | None = None,
        return_labels: bool = True,
        transform: Transform = Identity(),
    ) -> None:
        self.data: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None

        self.name: str = name
        self.split: Optional[str] = split
        self.save_path: Optional[Path]
        if path is None:
            self.save_path = None
        elif isinstance(path, str):
            self.save_path = Path(path)
        elif isinstance(path, Path):
            self.save_path = path
        else:
            raise TypeError
        self.return_labels: bool = return_labels

        super().__init__(
            transform=transform,
        )

    def _get_item(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.data is None:
            raise ValueError("The data is not loaded. Please load the data before using the dataset.")

        if self.return_labels is True:
            if self.targets is None:
                raise ValueError("The targets is not loaded. Please load the data before using the dataset.")
            return self.data[idx], self.targets[idx]
        else:
            return self.data[idx]

    def __len__(self) -> int:
        if self.data is None:
            raise ValueError("The data is not loaded. Please load the data before using the dataset.")

        return len(self.data)

    def _prepare(self) -> None:
        # replace with download, but not all datasets are downloadable with the method
        load_classification(name=self.name, split=self.split, extract_path=self.save_path)

    def _load(self) -> None:
        data: np.ndarray
        targets: np.ndarray
        data, targets = load_classification(name=self.name, split=self.split, extract_path=self.save_path)

        transform: Compose = Compose([ToTorchTensor(), LabelTransform()])
        transform.fit(data, targets)

        self.data, self.targets = transform(data, targets)
