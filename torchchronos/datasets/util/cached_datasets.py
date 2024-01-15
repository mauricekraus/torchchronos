import os, json
from pathlib import Path

import numpy as np

from .prepareable_dataset import PrepareableDataset
from ...transforms.base import Transform
from ...transforms.transforms import LabelTransform, Identity


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

# TODO: CachedDataset?
class CachedDataset(PrepareableDataset):
    def __init__(
        self,
        name: str,
        split: Optional[str] = None,
        save_path: Optional[Path] = None,
        return_labels: bool = True,
        pre_transform: Transform = Identity(),
        post_transform: Transform = Identity()
    ) -> None:
        self.name = name
        self.split = split
        self.save_path = (save_path or Path(".cache/torchchronos/data")) / name #TODO: save_path in cache_dir umbenennen
        self._np_path = self.save_path / (name + ".npz")
        self._json_path = self.save_path / (name + ".json")
        self.meta_data: Optional[Dict] = None
        self.is_prepared = False

        if self._is_dataset_prepared():
            self._load_meta_data()

        super().__init__(transform=post_transform)

        self.return_labels = return_labels
        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def _is_dataset_prepared(self) -> bool:
        return os.path.exists(self._np_path) and os.path.exists(self._json_path)

    def _load_meta_data(self) -> None:
        self.meta_data = json.load(open(self._json_path, "r"))
        self.is_prepared = True

    def _prepare(self) -> None:
        data = self._load_dataset()

        prepared_data = self._process_data(data)

        self._create_metadata(prepared_data)

        # 5. Save it
        self._save_data(prepared_data)

        # 6. Remove temp folder
        # shutil.rmtree(extract_path)

    def _load_dataset(self):
        return self._get_data()

    def _process_data(self, data):
        # Process data with labels
        try:
            X_train, Y_train, X_test, Y_test = data
        except ValueError:
            raise ValueError("The method _get_data must return 4 values: X_train, Y_train, X_test, Y_test"
                             "If the dataset does not have targets, return None instead of Y_train and Y_test.")

        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)

        self.pre_transform.fit(X, Y)
        X, Y = self.pre_transform.transform(X, Y)
        X_train, Y_train = self.pre_transform.transform(X_train, Y_train)
        X_test, Y_test = self.pre_transform.transform(X_test, Y_test)

        return X, Y, X_train, Y_train, X_test, Y_test
    

    def _create_metadata(self, data):
        # Create metadata
        X, Y, X_train, Y_train, X_test, Y_test = data
    
        meta_data = {
            "num_features": X.shape[1],
            "num_samples": X.shape[0],
            "num_train_samples": X_train.shape[0],
            "num_test_samples": X_test.shape[0],
            "length": X.shape[2],
        }

        self.meta_data = meta_data

    def _save_data(self, data):
        X, Y, X_train, Y_train, X_test, Y_test = data
        # Save data
        os.makedirs(self.save_path, exist_ok=True)

        if Y is not None:
            np.savez(self._np_path, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
        else:
            np.savez(self._np_path, X_train=X_train, X_test=X_test)
        json.dump(self.meta_data, open(self._json_path, "w"))

    def _load(self) -> None:
        data = np.load(self._np_path)
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
        print(self.data.shape)

    def _get_item(self, idx: int) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        print(self.data[idx].shape)
        if (self.targets is not None) and self.return_labels:
            return self.data[idx], self.targets[idx]
        else:
            return self.data[idx], None

    def __len__(self) -> int:
        return self.meta_data["num_samples"]
    
    # TODO: Abstractmethod?
    def _get_data():
        pass



class ClassificationDataset(CachedDataset):
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
        
        pre_transform = LabelTransform()
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

