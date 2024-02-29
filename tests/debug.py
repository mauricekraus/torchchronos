import torch
import numpy as np
from pathlib import Path
from torchchronos.transforms import Compose, Shift
from torchchronos.transforms.basic_transforms import Normalize
from torchchronos.datasets.cached_datasets import CachedDataset
from torchchronos.datasets.aeon_datasets import AeonClassificationDataset
from torchchronos.datasets.utils import save_dataset

dataset = AeonClassificationDataset(name="GunPoint")
dataset.prepare()
dataset.load()
transform = Compose([Normalize(), Shift(10)])
transform.fit(dataset)
transformed_dataset = transform(dataset)
save_dataset(transformed_dataset, "test")

c = CachedDataset("test")
c.prepare()
c.load()
print(c[:][0].shape)    