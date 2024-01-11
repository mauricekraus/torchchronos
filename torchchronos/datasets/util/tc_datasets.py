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

from .prepareable_dataset import PrepareableDataset
from ...transforms.base import Compose, Transform
from ...transforms.transforms import TransformLabels


"""
This class is a wrapper around the datasets from the aeon library.
It is used to make the datasets compatible with the torchchronos library and wrapps them into a PrepareableDataset.
The datasets are downloaded and prepared when the prepare method is called.
The datasets are loaded into memory when the load method is called.
The labels are transformed from strings with arbitrary values to integers starting from 0.
The meta_data dict contains the following keys:
    - num_features: The number of features in the dataset.
    - num_samples: The number of samples in the dataset.
    - num_train_samples: The number of samples in the training set.
    - num_test_samples: The number of samples in the test set.
    - length: The length of the time series in the dataset.
    - equal_samples_per_class: Whether the dataset has equal samples per class.
    - labels: A dict that maps the labels to integers.
    - num_classes: The number of classes in the dataset.
and is loaded in the constructor, if the dataset is already prepared.

It is possible to load all datasets that are available through the methods load_classification, load_regression and load_forecasting.
The has_y parameter is used to indicate whether the dataset has labels or not. 
The return_labels parameter is used to indicate whether the labels should be returned when the dataset is used.

This class is mainly used to create simple Dataset classes that are used in the experiments. Some examples can be found in the datasets.datasets file.
"""


class TCDataset(PrepareableDataset):
    """
    A dataset class for handling Aeon datasets.

    Args:
        name (str): The name of the dataset.
        split (str | None, optional): The split of the dataset ("train", "test" or None). Defaults to None.
        save_path (str | None, optional): The path to save the dataset. Defaults to None.
        prepare (bool, optional): Whether to prepare the dataset. Defaults to False.
        load (bool, optional): Whether to load the dataset. Defaults to False.
        has_y (bool, optional): Whether the dataset has labels. Defaults to True.
        return_labels (bool, optional): Whether to return labels when accessing the dataset. Defaults to True.
        use_cache (bool, optional): Whether to use the cached dataset. Defaults to True.
    """

    def __init__(
        self,
        name: str,
        load_method: callable,
        load_method_args: dict = {},
        split: str | None = None,
        save_path: Path | None = None,
        has_y: bool = True,
        return_labels: bool = True,
        pre_transform: Transform | None = None,
        post_transform: Transform | None = None,
        use_cache: bool = True,
    ) -> None:
        self.name = name
        self.split = split
        if save_path is None:
            save_path = Path(".cache/data/torchchronos")

        self.save_path = save_path / name
        self._np_path = self.save_path / (self.name + ".npz")
        self._json_path = self.save_path / (self.name + ".json")
        self.meta_data: dict | None = None

        if os.path.exists(self._np_path) and os.path.exists(self._json_path):
            self.meta_data = json.load(open(self._json_path, "r"))
            self.is_prepared = True

        super().__init__(transform=post_transform)

        self.X: np.ndarray | None = None
        self.has_y: bool = has_y
        if self.has_y:
            self.y: np.ndarray | None = None
            self.return_labels = return_labels

        self.load_method = load_method
        self.load_method_args = load_method_args
        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def _prepare(self) -> None:
        # 1. Check if dataset is already downloaded

        # 2. If not, download it
        extract_path = f".cache/temp"

        data = self.load_method(**self.load_method_args)
        if self.has_y:
            try:
                X, Y, X_train, Y_train, X_test, Y_test = data
            except ValueError:
                raise ValueError(
                    "The dataset does not have a train and test split, but has_y is set to True."
                )
        else:
            X, X_train, X_test = data

        # 2.1 transform data
        if self.pre_transform is not None:
            if self.has_y:
                self.pre_transform.fit(X, Y)
                X, Y = self.pre_transform.transform(X, Y)
            else:
                self.pre_transform.fit(X)
                X = self.pre_transform.transform(X)

        # 4. create Metadata
        # calculations if dataset has equal samples per class
        if self.has_y:
            unique_values, counts = np.unique(Y, return_counts=True)
            if np.all(counts == counts[0]):
                equal_samples_per_class = True
            else:
                equal_samples_per_class = False

        meta_data = {
            "num_features": X.shape[1],
            "num_samples": X.shape[0],
            "num_train_samples": X_train.shape[0],
            "num_test_samples": X_test.shape[0],
            "length": X.shape[2],
            "equal_samples_per_class": equal_samples_per_class,
        }

        self.meta_data = meta_data
        # 5. save it
        os.makedirs(self.save_path, exist_ok=True)
        if self.has_y:
            np.savez(
                self._np_path,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
            )
        else:
            np.savez(self._np_path, X_train=X_train, X_test=X_test)
        json.dump(meta_data, open(self._json_path, "w"))

        # 6. remove temp folder
        # shutil.rmtree(extract_path)

    def _load(self) -> None:
        data = np.load(self._np_path)

        if self.split == "train":
            self.X = data["X_train"]
            if self.has_y:
                self.y = data["Y_train"]
        elif self.split == "test":
            self.X = data["X_test"]
            if self.has_y:
                self.y = data["Y_test"]
        else:
            self.X = np.concatenate((data["X_train"], data["X_test"]), axis=0)
            if self.has_y:
                self.y = np.concatenate((data["Y_train"], data["Y_test"]), axis=0)

    def getItem(self, idx: int) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        if self.has_y and self.return_labels:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

    def __len__(self) -> int:
        return self.meta_data["num_samples"]


class ClassificationDataset(TCDataset):
    def __init__(
        self,
        name: str,
        load_method: callable,
        load_method_args: dict = {},
        split: str | None = None,
        save_path: Path | None = None,
        prepare: bool = False,
        load: bool = False,
        return_labels: bool = True,
        pre_transform: Transform | None = None,
        post_transform: Transform | None = None,
        use_cache: bool = True,
    ) -> None:
        super().__init__(
            name,
            load_method,
            load_method_args,
            split,
            save_path,
            prepare,
            load,
            True,
            return_labels,
            pre_transform,
            post_transform,
            use_cache,
        )


class RegressionDataset(TCDataset):
    def __init__(
        self,
        name: str,
        load_method: callable,
        load_method_args: dict = {},
        split: str | None = None,
        save_path: Path | None = None,
        prepare: bool = False,
        load: bool = False,
        pre_transform: Transform | None = None,
        post_transform: Transform | None = None,
        use_cache: bool = True,
    ) -> None:
        
        super().__init__(
            name,
            load_method,
            load_method_args,
            split,
            save_path,
            prepare,
            load,
            False,
            False,
            pre_transform,
            post_transform,
            use_cache,
        )


class ForecastingDataset(TCDataset):
    pass
