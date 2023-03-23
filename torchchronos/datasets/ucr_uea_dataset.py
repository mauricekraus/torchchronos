from pathlib import Path
from typing import Callable, Optional, Union
import pandas as pd
from sktime.datasets import load_UCR_UEA_dataset
import torch
from torch.utils.data import Dataset

from ..transforms import Compose

from ..utils import get_project_root


class UCRUEADataset(Dataset):
    def __init__(
        self,
        ds_name: str,
        path: Path,
        transform: Optional[
            Union[Callable[[torch.Tensor], torch.Tensor], Compose]
        ] = None,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.xs, self.ys = load_UCR_UEA_dataset(
            ds_name,
            extract_path=get_project_root() / path,
            return_type="numpy3d",
        )
        self.xs = torch.tensor(self.xs, dtype=torch.float32).transpose(1, 2)
        if self.transform is not None:
            self.transform = self.transform.fit(self.xs)

        if self.ys.dtype == "U2" or self.ys.dtype == "<U1":
            # convert string labels to int
            factorized_y = pd.factorize(self.ys, sort=True)
            self.y_labels = factorized_y[1]
            self.ys = factorized_y[0]

        self.ys = torch.tensor(self.ys, dtype=torch.float32)

        if len(self.ys.shape) == 1:
            self.ys = self.ys.unsqueeze(-1)

    def label_from_float_index(self, index: float) -> str:
        # check if float index is int
        if index.is_integer():
            return self.y_labels[int(index)]
        else:
            raise ValueError("Index is not an integer.")

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.xs[index]), self.ys[index]
        return self.xs[index], self.ys[index]
