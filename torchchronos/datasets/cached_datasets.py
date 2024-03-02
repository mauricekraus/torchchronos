from pathlib import Path
from typing import Any, Optional
import os

import numpy as np
import torch

from ..transforms.base_transforms import Transform
from ..transforms.basic_transforms import Identity
from ..transforms.format_conversion_transforms import ToTorchTensor

# from .prepareable_dataset import PrepareableDataset
from . import prepareable_dataset as pd


class CachedDataset(pd.PrepareableDataset):
    def __init__(
        self,
        name: str,
        save_path: Path | str = Path(".cache/torchchronos/datasets"),
        return_labels: bool = True,
        transform: Transform = Identity(),
    ) -> None:
        self.name: str = name
        self.data: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None

        self.return_labels: bool = return_labels
        self.path: Path
        if isinstance(save_path, str):
            self.path = Path(save_path)
        elif isinstance(save_path, Path):
            self.path = save_path
        else:
            raise TypeError

        super().__init__(transform=transform)

    def _get_data(self) -> tuple[np.ndarray, None] | tuple[np.ndarray, np.ndarray]:
        data_dict = np.load(self.path / f"{self.name}.npz", mmap_mode="r")
        data: np.ndarray = data_dict["data"]
        if "targets" in data_dict.files:
            targets: np.ndarray = data_dict["targets"]
            return data, targets
        else:
            return data, None

    def _prepare(self) -> None:
        if os.path.exists(self.path / f"{self.name}.npz") is False:
            raise FileNotFoundError

        # data, targets = self._get_data()
        # TODO: Maybe more checks on the data?

    def _load(self) -> None:
        data: np.ndarray
        targets: Optional[np.ndarray]
        data, targets = self._get_data()
        self.data, self.targets = ToTorchTensor()(data, targets)
        self.transforms.fit(self.data, self.targets)

    def _get_item(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.data is None:
            raise Exception("The data has to be loaded first, call prepare and load first.")

        if self.return_labels and self.targets is None:
            raise Exception("The targets have to be loaded first, call prepare and load first.")

        if self.return_labels:
            if self.return_labels and self.targets is None:
                raise Exception("The targets have to be loaded first, call prepare and load first.")

            return self.data[index], self.targets[index]
        else:
            return self.data[index]

    def __len__(self) -> int:
        if self.data is None:
            raise Exception("The data has to be loaded first, call prepare and load first.")

        return len(self.data)
