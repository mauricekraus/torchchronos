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
    """
    A dataset class for loading cached data.


    Attributes:
        name (str): The name of the dataset.
        data (Optional[torch.Tensor]): The loaded data.
        targets (Optional[torch.Tensor]): The loaded targets.
        return_labels (bool): Whether to return labels along with the data.
        path (Path): The path to save the cached data.


    """

    def __init__(
        self,
        name: str,
        save_path: Path | str = Path(".cache/torchchronos/datasets"),
        return_labels: bool = True,
        transform: Transform = Identity(),
    ) -> None:
        """
        Initializes a new instance of the CachedDataset class.

        Args:
            name (str): The name of the dataset.
            save_path (Path | str, optional): The path to save the cached data. Defaults to ".cache/torchchronos/datasets".
            return_labels (bool, optional): Whether to return labels along with the data. Defaults to True.
            transform (Transform, optional): The data transformation to apply. Defaults to Identity().

        Raises:
            TypeError: If the save_path is not a string or a Path object.
            FileNotFoundError: If the cached data file does not exist.
        """
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
        """
        Load the data from the cached file.

        Returns:
            tuple[np.ndarray, None]: The loaded data. Without targets.
            tuple[np.ndarray, np.ndarray]: The loaded data and targets (if available).

        """
        data_dict = np.load(self.path / f"{self.name}.npz", mmap_mode="r")
        data: np.ndarray = data_dict["data"]
        if "targets" in data_dict.files:
            targets: np.ndarray = data_dict["targets"]
            return data, targets
        else:
            return data, None

    def _prepare(self) -> None:
        """
        Prepare the dataset for loading.

        Raises:
            FileNotFoundError: If the cached data file does not exist.

        """
        if os.path.exists(self.path / f"{self.name}.npz") is False:
            raise FileNotFoundError

        # data, targets = self._get_data()
        # TODO: Maybe more checks on the data?

    def _load(self) -> None:
        """
        Load the data and targets into memory.

        """
        data: np.ndarray
        targets: Optional[np.ndarray]
        data, targets = self._get_data()
        self.data, self.targets = ToTorchTensor()(data, targets)
        self.transforms.fit(self.data, self.targets)

    def _get_item(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Get a specific item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            torch.Tensor: The item of the dataset, without the label.
            tuple[torch.Tensor, torch.Tensor]: The data item or a tuple of data and targets.

        Raises:
            Exception: If the data has not been loaded yet.
            Exception: If the targets have not been loaded yet.

        """
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
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.

        Raises:
            Exception: If the data has not been loaded yet.

        """
        if self.data is None:
            raise Exception("The data has to be loaded first, call prepare and load first.")

        return len(self.data)
