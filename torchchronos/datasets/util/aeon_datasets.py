import os, json, shutil
from pathlib import Path

import numpy as np
from aeon.datasets._data_loaders import load_classification, load_regression, load_forecasting
from aeon.datasets.tsc_data_lists import multivariate, univariate
from aeon.datasets.tser_data_lists import tser_all
from aeon.datasets.tsf_data_lists import tsf_all


from .prepareable_dataset import PrepareableDataset


"""
Classes for loading datasets from Aeon.
Regression, Classification classes needed.
Use PretrainDataset as a template.
"""


class AeonDataset(PrepareableDataset):
    def __init__(
        self,
        name: str,
        split: str | None = None,
        save_path: str | None = None,
        prepare: bool = False,
        load: bool = False,
        has_y = True,
        return_labels: bool = True,
        use_cache: bool = True
    ) -> None:
        
        self.name = name
        self.split = split
        if save_path is None:
            self.save_path = Path(".cache/data/torchchronos") / name

        self._np_path = self.save_path / (self.name + ".npz")
        self._json_path = self.save_path / (self.name + ".json")
        self.meta_data: dict | None = None

        if use_cache:
            if os.path.exists(self._np_path) and os.path.exists(self._json_path):
                self.meta_data = json.load(open(self._json_path, "r"))
                self.is_prepared = True

        super().__init__(prepare, load)
        
        self.X: np.ndarray | None = None
        self.has_y: bool = has_y
        if self.has_y:
            self.y: np.ndarray | None = None
            self.return_labels = return_labels

    def _prepare(self) -> None:
        # 1. Check if dataset is already downloaded
        
        # 2. If not, download it
        extract_path = f".cache/temp"

        load_method = None
        if self.name in univariate or self.name in multivariate:
            load_method = load_classification
            self.has_y = True
        elif self.name in tser_all:
            load_method = load_regression
            self.has_y = True
        elif self.name in tsf_all:
            load_method = load_forecasting
            self.has_y = False

        if self.has_y:
            X, Y, meta = load_method(name=self.name, split=None, extract_path=extract_path, return_metadata=True)
            X_train, Y_train = load_method(name=self.name, split="train", extract_path=extract_path, return_metadata=False)
            X_test, Y_test = load_method(name=self.name, split="test", extract_path=extract_path, return_metadata=False)
        else:
            X, meta = load_method(name=self.name, split=None, extract_path=extract_path, return_metadata=True)
            X_train = load_method(name=self.name, split="train", extract_path=extract_path, return_metadata=False)
            X_test = load_method(name=self.name, split="test", extract_path=extract_path, return_metadata=False)

        # 3. transform labels
        if self.has_y and self.return_labels:
            label_map = self._get_label_map(Y)
            for i in range(len(Y_train)):
                Y_train[i] = label_map[Y_train[i]]
            for i in range(len(Y_test)):
                Y_test[i] = label_map[Y_test[i]]
        
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
            "equal_samples_per_class": equal_samples_per_class}
        if self.has_y:
            temp_dict = {
                "labels": label_map,
                "num_classes": len(unique_values.tolist())}
            temp_dict.update(meta_data)
            meta_data = temp_dict

        # 5. save it
        os.makedirs(self.save_path, exist_ok=True)
        if self.has_y:
            np.savez_compressed(self._np_path, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
        else:
            np.savez_compressed(self._np_path, X_train=X_train, X_test=X_test)
        json.dump(meta_data, open(self._json_path, "w"))

        # 6. remove temp folder
        shutil.rmtree(extract_path)

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


    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        super().__getitem__(idx)
        if self.has_y:
            if self.return_labels:
                return self.X[idx], self.y[idx]
            else:
                return self.X[idx]
        else:
            return self.X[idx]


    def __len__(self) -> int:
        return self.meta_data["num_samples"]

    def _get_label_map(self, y) -> dict[int, int]:
        labels = np.unique(y)
        labels = np.sort(labels)
        label_map = {label: i for i, label in enumerate(labels)}

        return label_map
