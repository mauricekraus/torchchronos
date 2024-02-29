from pathlib import Path
from typing import Optional

from aeon.datasets._data_loaders import load_classification

from ...transforms.base_transforms import Transform, Compose
from ...transforms.representation_transformations import LabelTransform
from ...transforms.basic_transforms import Identity
from ...transforms.format_conversion_transforms import ToTorchTensor

from .cached_datasets import CachedDataset


class AeonClassificationDataset(CachedDataset):
    def __init__(
        self,
        name: str,
        split: Optional[str] = None,
        save_path: Optional[Path] = None,
        return_labels: bool = True,
        pre_transform: Transform = Identity(),
        transform: Transform = Identity(),
    ) -> None:
        label_transform = Compose([ToTorchTensor(), LabelTransform()])
        if pre_transform is None:
            pre_transform = label_transform
        else:
            label_transform += pre_transform
            pre_transform = label_transform

        super().__init__(
            name=name,
            split=split,
            save_path=save_path,
            return_labels=return_labels,
            pre_transform=pre_transform,
            transform=transform,
        )

    def _get_data(self):
        X_train, Y_train = load_classification(
            name=self.name, split="train", return_metadata=False
        )
        X_test, Y_test = load_classification(
            name=self.name, split="test", return_metadata=False
        )
        return X_train, Y_train, X_test, Y_test
