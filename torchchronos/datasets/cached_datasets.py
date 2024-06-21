"""Class for loading and using Cached Datasets."""

import os
from pathlib import Path

import numpy as np
import torch

from ..transforms import Identity, ToTorchTensor, Transform
from .prepareable_dataset import PrepareableDataset


class CachedDataset(PrepareableDataset):
    """A dataset class for loading cached data.

    This class is a PrepareableDataset and therefore has a prepare and a load step.
    In the prepare step, it is checked whether the path to the data exists.
    In the load step, the data is loaded into the memory and transformed.
    Currently only numpy files are supported.

    Args:
        name: The name of the dataset.
        save_path: The path to save the cached data.
        return_labels: Whether to return labels along with the data. Defaults to True.
        transform: The data transformation to apply. Defaults to Identity().

    Examples:
        Load a dataset from a cache.

        >>> from torchchronos.datasets import AeonClassificationDataset
        >>> from torchchronos.datasets.utils import save_dataset
        >>> #
        >>> datasets = AeonClassificationDataset(name="GunPoint")

    """

    def __init__(
        self,
        name: str,
        save_path: Path | str = Path(".cache/torchchronos/datasets"),
        return_labels: bool = True,
        transform: Transform = Identity(),
    ) -> None:
        self.name: str = name
        self.data: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None

        self.return_labels: bool = return_labels
        self.path: Path
        if isinstance(save_path, str):
            self.path = Path(save_path)
        elif isinstance(save_path, Path):
            self.path = save_path
        else:
            raise TypeError

        super().__init__(transform=transform)

    def _get_data(self) -> tuple[torch.Tensor, None] | tuple[torch.Tensor, torch.Tensor]:
        """Load the data from the cached file.

        Returns:
            tuple: The loaded data. Without targets.
            tuple: The loaded data and targets (if available).

        """
        data_dict = np.load(self.path / f"{self.name}.npz", mmap_mode="r")
        data: np.ndarray = data_dict["data"]
        if "targets" in data_dict.files:
            targets: np.ndarray = data_dict["targets"]
            return torch.from_numpy(data), torch.from_numpy(targets)
        else:
            return torch.from_numpy(data), None

    def _prepare(self) -> None:
        """Prepare the dataset for loading.

        Raises:
            FileNotFoundError: If the cached data file does not exist.

        """
        if os.path.exists(self.path / f"{self.name}.npz") is False:
            raise FileNotFoundError

    def _load(self) -> None:
        """Load the data and targets into memory."""
        data: torch.Tensor
        targets: torch.Tensor | None
        self.data, self.targets = self._get_data()
        self.transforms.fit(self.data, self.targets)

    def _get_item(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Get a specific item from the dataset.

        Args:
            index: The index of the item to retrieve.

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
        """Get the length of the dataset.

        Returns:
            int: The length of the dataset.

        Raises:
            Exception: If the data has not been loaded yet.

        """
        if self.data is None:
            raise Exception("The data has to be loaded first, call prepare and load first.")

        return len(self.data)
