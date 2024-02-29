from pathlib import Path
from typing import Optional

import torch
from aeon.datasets._data_loaders import load_classification

from ..transforms.base_transforms import Transform, Compose
from ..transforms.representation_transformations import LabelTransform
from ..transforms.basic_transforms import Identity
from ..transforms.format_conversion_transforms import ToTorchTensor

from .prepareable_dataset import PrepareableDataset


class AeonClassificationDataset(PrepareableDataset):
    def __init__(
        self,
        name: str,
        split: Optional[str] = None,
        save_path: Optional[Path] = None,
        return_labels: bool = True,
        transform: Transform = Identity(),
    ) -> None:
        self.data: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None

        self.name = name
        self.split = split
        self.save_path = save_path
        self.return_labels = return_labels

        super().__init__(
            transform=transform,
        )

    def _get_item(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.return_labels is True:
            return self.data[idx], self.targets[idx]
        else:
            return self.data[idx]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _prepare(self) -> None:
        # replace with download, but not all datasets are downloadable with the method
        load_classification(name=self.name, split=self.split, extract_path=self.save_path)

    def _load(self) -> None:
        data, targets = load_classification(name=self.name, split=self.split, extract_path=self.save_path)

        transform = Compose([ToTorchTensor(), LabelTransform()])
        transform.fit(data, targets)

        self.data, self.targets = transform.transform(data, targets)

