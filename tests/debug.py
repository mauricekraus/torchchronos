import torch
from torchchronos.transforms.base_transforms import Transform, Compose

from torchchronos.transforms.format_conversion_transforms import ToTorchTensor
from torchchronos.transforms.transforms import Shift
from torchchronos.datasets.util.aeon_datasets import AeonClassificationDataset

data = AeonClassificationDataset("GunPoint", split=None, return_labels=False)
data.prepare()
data.load()
data = ToTorchTensor().transform(data)
data = Shift(10.0).transform(data)

print(data[0])