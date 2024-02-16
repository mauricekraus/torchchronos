from typing import Optional
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, data: torch.Tensor, targets:Optional[torch.Tensor]=None) -> None:
        if (targets is not None) and (len(data) != len(targets)):
            raise ValueError("Data and target should have the same length")
        self.data = data
        self.targets = targets

    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, None]:
        if self.targets is None:
            return self.data[idx], None
        else:
            return self.data[idx], self.targets[idx]
        
    def __len__(self) -> int:
        return len(self.data)
        
