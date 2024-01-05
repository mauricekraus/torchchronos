import os, json, shutil

import numpy as np
from aeon.datasets._data_loaders import load_classification

from prepareable_dataset import PrepareableDataset
from pathlib import Path

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

        self._np_path = self.path / self.name / ".npy"
        self._json_path = self.path / self.name / ".json"
        self.meta_data: dict | None = None

        if use_cache:
            if os.path.exists(self.np_path) and os.path.exists(self.json_path):
                self.meta_data = json.load(open(self.json_path, "r"))
                self.is_prepared = True

        super().__init__(prepare, load)
        
        self.X: np.ndarray | None = None
        if has_y:
            self.y: np.ndarray | None = None
            self.return_labels = return_labels

    def _prepare(self) -> None:
        # 1. Check if dataset is already downloaded
        if os.path.exists(self.np_path) and os.path.exists(self.json_path):
            return
        
        # 2. If not, download it
        extract_path = f".cache/temp"
        if self.has_y:
            X, Y, meta = load_classification(name=self.name, split=None, extract_path=extract_path, return_metadata=True)
            X_train, Y_train = load_classification(name=self.name, split="train", extract_path=extract_path, return_metadata=False)
            X_test, Y_test = load_classification(name=self.name, split="test", extract_path=extract_path, return_metadata=False)
        else:
            X, meta = load_classification(name=self.name, split=None, extract_path=extract_path, return_metadata=True)
            X_train = load_classification(name=self.name, split="train", extract_path=extract_path, return_metadata=False)
            X_test = load_classification(name=self.name, split="test", extract_path=extract_path, return_metadata=False)

        # 3. transform labels
        if self.has_y and self.return_labels:
            label_map = self.update_labels()
            for y in Y_train:
                y = label_map[y]
            for y in Y_test:
                y = label_map[y]
        
        # 4. create Metadata
        # calculations if dataset has equal samples per class
        if self.has_y:
            unique_values = counts = np.unique(Y, return_counts=True)[1]
            if np.all(counts == counts[0]):
                equal_samples_per_class = True
            else:
                equal_samples_per_class = False

        meta_data = {
            "labels": label_map,
            "num_features": X.shape[1],
            "num_samples": X.shape[0], 
            "num_train_samples": X_train.shape[0],
            "num_test_samples": X_test.shape[0],
            "length": X.shape[2], 
            "equal_samples_per_class": equal_samples_per_class}
        if self.has_y:
            meta_data["num_classes"] = unique_values
        
        # 5. save it
        os.makedirs(self.save_path, exist_ok=True)
        if self.has_y:
            np.savez_compressed(self.np_path, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
        else:
            np.savez_compressed(self.np_path, X_train=X_train, X_test=X_test)
        json.dump(meta_data, open(self.json_path, "w"))

        # 6. remove temp folder
        shutil.rmtree(extract_path)

    def _load(self) -> None:
        data = np.load(self.np_path)

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

    def get_label_map(self) -> dict[int, int]:
        labels = np.unique(self.y)
        labels = np.sort(labels)
        label_map = {label: i for i, label in enumerate(labels)}

        return label_map



