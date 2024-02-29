import torch
import pytest

from torchchronos.datasets.base_dataset import BaseDataset

def test_base_dataset():
    data = torch.randn(10, 1, 100)
    target = torch.randint(0, 2, (10,))

    dataset = BaseDataset(data, target)
    assert len(dataset) == 10
    assert len(dataset[0]) == 2
    assert dataset[0][0].shape == torch.Size([1, 100])
    assert dataset[0][1].shape == torch.Size([])

    dataset = BaseDataset(data)
    assert len(dataset) == 10
    assert len(dataset[0]) == 1
    assert dataset[0].shape == torch.Size([1, 100])
