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

from .tc_datasets import ClassificationDataset, RegressionDataset, ForecastingDataset
from ...transforms.base import Compose, Transform
from ...transforms.transforms import TransformLabels


class AeonClassificationDataset(ClassificationDataset):
    def __init__(
        self,
        name: str,
        split: str | None = None,
        save_path: Path | None = None,
        prepare: bool = False,
        load: bool = False,
        return_labels: bool = True,
        use_cache: bool = True,
        pre_transform: Transform | None = None,
        post_transform: Transform | None = None,
    ) -> None:
        def get_data():
            X, Y = load_classification(name=name, split=None, return_metadata=False)
            X_train, Y_train = load_classification(
                name=name, split="train", return_metadata=False
            )
            X_test, Y_test = load_classification(
                name=name, split="test", return_metadata=False
            )
            return X, Y, X_train, Y_train, X_test, Y_test

        if pre_transform is None:
            transform = TransformLabels
        elif isinstance(transform, Compose):
            transform.append(TransformLabels)
        else:
            transform = Compose([TransformLabels, transform])

        super().__init__(
            name=name,
            load_method=get_data,
            load_method_args={},
            split=split,
            save_path=save_path,
            prepare=prepare,
            load=load,
            return_labels=return_labels,
            use_cache=use_cache,
            pre_transform=pre_transform,
            post_transform=post_transform,
        )


class AeonRegressionDataset(RegressionDataset):
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
    ) -> None:
        def get_data():
            X = load_regression(name=name, split=None, return_metadata=False)
            X_train = load_regression(
                name=name, split="train", return_metadata=False
            )
            X_test = load_regression(
                name=name, split="test", return_metadata=False
            )
            return X, X_train, X_test
        super().__init__(
            name = name,
            load_method=get_data,
            split=split,
            save_path=save_path,
            prepare=prepare,
            load=load,
            pre_transform=pre_transform,
            post_transform=post_transform,
        )


class AeonForecastingDataset(ForecastingDataset):
    pass
