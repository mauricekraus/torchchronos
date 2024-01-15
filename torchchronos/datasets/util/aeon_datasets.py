import os, json, shutil
from pathlib import Path

import numpy as np
from aeon.datasets._data_loaders import (
    load_classification,
    load_regression,
    load_forecasting,
)
from aeon.datasets.tsc_data_lists import multivariate, univariate
from aeon.datasets.tser_data_lists import tser_all
from aeon.datasets.tsf_data_lists import tsf_all

from .cached_datasets import CachedDataset
from ...transforms.base import Transform
from ...transforms.transforms import LabelTransform, Identity


class AeonClassificationDataset(CachedDataset):
    def __init__(
        self,
        name: str,
        split: str | None = None,
        save_path: Path | None = None,
        pre_transform: Transform  = Identity(),
        post_transform: Transform  = Identity(),
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


