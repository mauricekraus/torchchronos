from pathlib import Path
from typing import Any
import os

import numpy as np

from ..transforms.base_transforms import Transform
from ..transforms.basic_transforms import Identity
from ..transforms.format_conversion_transforms import ToTorchTensor

# from .prepareable_dataset import PrepareableDataset
from . import prepareable_dataset as pd


class CachedDataset(pd.PrepareableDataset):
    def __init__(
        self,
        name: str,
        path: Path | str = Path(".cache/torchchronos/datasets"),
        return_labels: bool = True,
        transform: Transform = Identity(),
    ) -> None:
        self.name = name

        self.return_labels: bool = return_labels
        if isinstance(path, str):
            self.path: Path = Path(path)
        else:
            self.path = path

        super().__init__(transform=transform)

    def _get_data(self):
        data_dict = np.load(self.path / f"{self.name}.npz", mmap_mode="r")
        data = data_dict["data"]
        print(data.shape)
        if "targets" in data_dict.files:
            targets = data_dict["targets"]
            return data, targets
        else:
            return data

    def _prepare(self) -> None:
        if os.path.exists(self.path / f"{self.name}.npz") is False:
            raise FileNotFoundError

        data, targets = self._get_data()
        # TODO: Maybe more checks on the data?

    def _load(self) -> None:
        data, targets = self._get_data()
        self.data, self.targets = ToTorchTensor()(data, targets)

    def _get_item(self, index: int) -> Any:
        if self.return_labels:
            return self.data[index], self.targets[index]
        else:
            return self.data[index], None

    def __len__(self) -> int:
        if not self.is_loaded:
            raise Exception

        return len(self.data)
