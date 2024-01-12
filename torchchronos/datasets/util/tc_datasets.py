import os, json
from pathlib import Path

import numpy as np

from .prepareable_dataset import PrepareableDataset
from ...transforms.base import Transform
from ... transforms.transforms import TransformLabels


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


from pathlib import Path
from typing import Callable, Dict, Optional

class TCDataset(PrepareableDataset):
    def __init__(
        self,
        name: str,
        load_method: Callable,
        load_method_args: Dict = {},
        split: Optional[str] = None,
        save_path: Optional[Path] = None,
        has_y: bool = True,
        return_labels: bool = True,
        pre_transform: Optional[Transform] = None,
        post_transform: Optional[Transform] = None,
    ) -> None:
        self.name = name
        self.split = split
        self.save_path = save_path or Path(".cache/data/torchchronos") / name
        self._np_path = self.save_path / (name + ".npz")
        self._json_path = self.save_path / (name + ".json")
        self.meta_data: Optional[Dict] = None
        self.is_prepared = False

        if self._is_dataset_prepared():
            self._load_meta_data()

        super().__init__(transform=post_transform, has_y=has_y)

        self.return_labels = return_labels
        self.load_method = load_method
        self.load_method_args = load_method_args
        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def _is_dataset_prepared(self) -> bool:
        return os.path.exists(self._np_path) and os.path.exists(self._json_path)

    def _load_meta_data(self) -> None:
        self.meta_data = json.load(open(self._json_path, "r"))
        self.is_prepared = True

    def _prepare(self) -> None:

        extract_path = ".cache/temp"

        data = self._load_dataset()
        if self.has_y:
            prepared_data = self._process_data_with_labels(data)
        else:
            prepared_data = self._process_data_without_labels(data)

        self._create_metadata(data)

        # 5. Save it
        self._save_data(prepared_data)

        # 6. Remove temp folder
        # shutil.rmtree(extract_path)

    def _load_dataset(self):
        return self.load_method(**self.load_method_args)

    def _process_data_with_labels(self, data):
        # Process data with labels
        try:
            X, Y, X_train, Y_train, X_test, Y_test = data
        except ValueError:
            raise ValueError("The dataset does not have a train and test split, but has_y is set to True.")

        if self.pre_transform is not None:
            self.pre_transform.fit(X, Y)
        X, Y = self._apply_pre_transform(X, Y)
        X_train, Y_train = self._apply_pre_transform(X_train, Y_train)
        X_test, Y_test = self._apply_pre_transform(X_test, Y_test)

        return X, Y, X_train, Y_train, X_test, Y_test
    
    def _process_data_without_labels(self, data):
        # Process data without labels
        X, X_train, X_test = data
        if self.pre_transform is not None:
            self.pre_transform.fit(X)
            X = self._apply_pre_transform(X)
        X_train = self._apply_pre_transform(X_train)
        X_test = self._apply_pre_transform(X_test)

        return X, X_train, X_test

    def _apply_pre_transform(self, X, Y=None):
        # Apply pre-transform if specified
        if self.has_y:
            if self.pre_transform is not None:
                X, Y = self.pre_transform.transform(X, Y)
            return X, Y

        else:
            if self.pre_transform is not None:
                X = self.pre_transform.transform(X)
            return X

    def _create_metadata(self, data):
        # Create metadata
        if self.has_y:
            X, Y, X_train, Y_train, X_test, Y_test = data
        else:
            X, X_train, X_test = data

        meta_data = {
            "num_features": X.shape[1],
            "num_samples": X.shape[0],
            "num_train_samples": X_train.shape[0],
            "num_test_samples": X_test.shape[0],
            "length": X.shape[2],
        }

        self.meta_data = meta_data

    def _save_data(self, data):
        # Save data
        os.makedirs(self.save_path, exist_ok=True)

        if self.has_y:
            X, Y, X_train, Y_train, X_test, Y_test = data
            np.savez(self._np_path, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
        else:
            X, X_train, X_test = data
            np.savez(self._np_path, X_train=X_train, X_test=X_test)
        json.dump(self.meta_data, open(self._json_path, "w"))

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
        return_labels: bool = True,
        pre_transform: Transform | None = None,
        post_transform: Transform | None = None,
    ) -> None:
        
        pre_transform = TransformLabels()
        super().__init__(
            name,
            load_method,
            load_method_args,
            split,
            save_path,
            True,
            return_labels,
            pre_transform,
            post_transform,
        )


class RegressionDataset(TCDataset):
    def __init__(
        self,
        name: str,
        load_method: callable,
        load_method_args: dict = {},
        split: str | None = None,
        save_path: Path | None = None,
        pre_transform: Transform | None = None,
        post_transform: Transform | None = None,
    ) -> None:
        
        super().__init__(
            name,
            load_method,
            load_method_args,
            split,
            save_path,
            False,
            False,
            pre_transform,
            post_transform,
        )


class ForecastingDataset(TCDataset):
    pass
