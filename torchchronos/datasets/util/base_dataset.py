from typing import Optional
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, data: torch.Tensor, targets:Optional[torch.Tensor]=None) -> None:
        self.data = data
        self.targets = targets

    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, None]:
        if self.targets is None:
            return self.data[idx], None
        else:
            return self.data[idx], self.targets[idx]
        
