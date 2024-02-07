from pathlib import Path
from typing import Optional

from aeon.datasets._data_loaders import load_classification

from ...transforms.base_transforms import Transform
from ...transforms.transforms import Identity
from ...transforms.representation_transformations import LabelTransform
from .cached_datasets import CachedDataset


class AeonClassificationDataset(CachedDataset):
    def __init__(
        self,
        name: str,
        split: Optional[str] = None,
        save_path: Optional[Path] = None,
        return_labels: bool = True,
        pre_transform: Transform = Identity(),
        post_transform: Transform = Identity(),
    ) -> None:
        label_transform = LabelTransform()
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
            post_transform=post_transform,
        )

    def _get_data(self):
        X_train, Y_train = load_classification(
            name=self.name, split="train", return_metadata=False
        )
        X_test, Y_test = load_classification(
            name=self.name, split="test", return_metadata=False
        )
        return X_train, Y_train, X_test, Y_test
