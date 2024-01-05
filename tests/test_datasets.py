from aeon.datasets import load_classification
from torchchronos.datasets.util.concat_dataset import ConcatDataset

import torch
from torch.utils.data import Dataset, TensorDataset


def test_concat_dataset():
    X = load_classification(name="GunPoint")
    dataset = X
    concat_dataset = ConcatDataset([dataset, dataset], [1.0, 1.0])
    assert len(concat_dataset) == 2 * len(dataset)


def test_shuffle_dataset():
    pass


test_concat_dataset()
