from typing import Optional

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    A base class for creating custom datasets in PyTorch.

    Attributes:
        data (torch.Tensor): The input data tensor.
        targets (Optional[torch.Tensor]): The target tensor (optional).

    """

    def __init__(self, data: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Initializes a new instance of the BaseDataset class.

        Args:
            data (torch.Tensor): The input data tensor.
            targets (torch.Tensor, optional): The target tensor. Defaults to None.

        Raises:
                ValueError: If the length of data and targets is not the same.

        """
        self.data = data
        self.targets = targets

        if targets is not None and len(data) != len(targets):
            raise ValueError("The length of data and targets must be the same.")

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single item or a tuple of data and target at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            torch.Tensor: The item of the dataset. If self.targets is None, only the data is returned.
            tuple[torch.Tensor, torch.Tensor]: The data item or a tuple of data and target.
        """
        if self.targets is not None:
            return self.data[idx], self.targets[idx]
        else:
            return self.data[idx]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)
