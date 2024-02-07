import torch

from torchchronos.transforms.format_conversion_transforms import ToTorchTensor

data = torch.randn(100, 100).reshape(100, 1, 100)
data = ToTorchTensor()(data)
