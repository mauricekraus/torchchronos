import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


from ...transforms.base_transforms import Transform
from ...transforms.basic_transforms import Identity
from ...transforms.structure_transforms import RemoveLabels
from .prepareable_dataset import PrepareableDataset

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


class CachedDataset(PrepareableDataset):
    def __init__(
        self,
        name: str,
        path: Path,
        return_labels: bool = True,
        transform: Transform = Identity(),
    ) -> None:
        self.name = name

        self.return_labels: bool = return_labels
        self.path: Path = path

        super().__init__(transform=transform)

    def _prepare(self) -> None:
        data = self._load_dataset()

        prepared_data = self._process_data(data)

        self._create_metadata(prepared_data)

        self._save_data(prepared_data)

    def _load_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            X_train, Y_train, X_test, Y_test = self._get_data()
        except ValueError:
            raise ValueError(
                "The method _get_data must return 4 values: X_train, Y_train, X_test, Y_test"
                "If the dataset does not have targets, return None instead of Y_train and Y_test."
            )
        return X_train, Y_train, X_test, Y_test

    def _process_data(self, data) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        np.ndarray,
        Optional[np.ndarray],
        np.ndarray,
        Optional[np.ndarray],
    ]:
        try:
            X_train, Y_train, X_test, Y_test = data
        except ValueError:
            raise ValueError(
                "The method _get_data must return 4 values: X_train, Y_train, X_test, Y_test"
                "If the dataset does not have targets, return None instead of Y_train and Y_test."
            )

        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)

        self.pre_transform.fit(X, Y)
        X, Y = self.pre_transform.transform(X, Y)
        X_train, Y_train = self.pre_transform.transform(X_train, Y_train)
        X_test, Y_test = self.pre_transform.transform(X_test, Y_test)

        return X, Y, X_train, Y_train, X_test, Y_test

    def _create_metadata(self, data) -> None:
        X, Y, X_train, Y_train, X_test, Y_test = data

        # TODO: Add more
        meta_data = {
            "num_features": X.shape[1],
            "num_samples": X.shape[0],
            "num_train_samples": X_train.shape[0],
            "num_test_samples": X_test.shape[0],
            "length": X.shape[
                2
            ],  # TODO: Check if this is correct 1 or 2, depends on dimensionality
        }

        self.meta_data = meta_data

    def _generate_file_name(self) -> str:
        save_string = f"{self.name}_{self.split}_{repr(self.pre_transform)}"

        sha1_hash = hashlib.sha1(save_string.encode("utf-8"))

        hash_hex = sha1_hash.hexdigest()

        file_uuid = uuid.UUID(hash_hex[:32])
        split = self.split or "all"
        file_name = f"{self.name}_{split}_{file_uuid}"
        return file_name

    def _save_data(self, data) -> None:
        X, Y, X_train, Y_train, X_test, Y_test = data
        os.makedirs(self.cache_dir, exist_ok=True)

        if Y is not None:
            np.savez(
                self.np_path,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
            )
        else:
            np.savez(self.np_path, X_train=X_train, X_test=X_test)
        json.dump(
            self.meta_data, open(self.json_path, "w")
        )  # TODO: close file after writing

    def _load(self) -> None:
        data = np.load(self.np_path)
        has_y = False

        if "Y_train" in data.files and "Y_test" in data.files:
            has_y = True

        if self.split == "train":
            self.data = data["X_train"]
            if has_y:
                self.targets = data["Y_train"]
        elif self.split == "test":
            self.data = data["X_test"]
            if has_y:
                self.targets = data["Y_test"]
        else:
            self.data = np.concatenate((data["X_train"], data["X_test"]), axis=0)
            if has_y:
                self.targets = np.concatenate((data["Y_train"], data["Y_test"]), axis=0)

        self.data = torch.from_numpy(self.data)
        if has_y:
            self.targets = torch.from_numpy(self.targets)

    def __len__(self) -> int:
        return self.meta_data["num_samples"]

    # TODO: Abstractmethod?
    def _get_data():
        pass
