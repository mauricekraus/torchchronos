import torch
import numpy as np
from torchchronos.transforms.basic_transforms import Normalize

data = torch.tensor([[1,2,3], [4,5,6]]).to(torch.float32).reshape(2, 1, 3)

transform = Normalize()

transform.fit(data)
print(transform.mean)
print(transform.std)