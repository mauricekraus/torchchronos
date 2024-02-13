import torch
from torchchronos.transforms.base_transforms import Transform, Compose

from torchchronos.transforms.format_conversion_transforms import ToTorchTensor
from torchchronos.transforms.structure_transforms import RemoveLabels, SlidingWindow
from torchchronos.transforms.transforms import Shift
from torchchronos.datasets.util.aeon_datasets import AeonClassificationDataset
from torch.utils.data import Dataset, TensorDataset


data = torch.arange(0, 300).reshape(3, 1, 100).float()
print(SlidingWindow(10, 5)(data).shape)

