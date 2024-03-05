"""Class for using the USR datasets from the Aeon library."""

from pathlib import Path

import numpy as np
import torch
from aeon.datasets._data_loaders import load_classification

from ..transforms import base_transforms as bt
from ..transforms.basic_transforms import Identity
from ..transforms.format_conversion_transforms import ToTorchTensor
from ..transforms.representation_transformations import LabelTransform
from .prepareable_dataset import PrepareableDataset


class AeonClassificationDataset(PrepareableDataset):
    """
    A Dataset class to load classification datasets from the Aeon library (UCR).

    It uses the PrepareableDataset class as a base class.
    In prepare the dataset is downloaded and extracted.
    In load the dataset is loaded and transformed.

    """

    def __init__(
        self,
        name: str,
        split: str | None = None,
        path: Path | str | None = None,
        return_labels: bool = True,
        transform: bt.Transform = Identity(),
    ) -> None:
        """
        Initialize a new instance of the AeonClassificationDataset class.

        Args:
            name (str): The name of the dataset.
            split (str, optional): The split of the dataset. Defaults to None.
            path (Path | str | None, optional): The path to save the dataset. Defaults to None.
            return_labels (bool, optional): Whether to return labels along with the data. Defaults to True.
            transform (Transform, optional): The data transformation to apply. Defaults to Identity().

        Raises
        ------
            TypeError: If the `path` argument is not of type `str`, `Path` or 'None'.
        """
        self.data: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None

        self.name: str = name
        self.split: str | None = split
        self.save_path: Path | None = None
        if path is None:
            self.save_path = None
        elif isinstance(path, str):
            self.save_path = Path(path)
        elif isinstance(path, Path):
            self.save_path = path
        else:
            raise TypeError("The 'path' argument must be of type 'str' or 'Path'.")

        self.return_labels: bool = return_labels

        super().__init__(
            transform=transform,
        )

    def _get_item(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns
        -------
            torch.Tensor: The item of the datset, without the label
            tuple[torch.Tensor, torch.Tensor]: The data item or a tuple of data and labels.

        Raises
        ------
            ValueError: If the data is not loaded or the targets are not loaded.
        """
        if self.data is None:
            raise ValueError("The data is not loaded. Please load the data before using the dataset.")

        if self.return_labels is True:
            if self.targets is None:
                raise ValueError("The targets is not loaded. Please load the data before using the dataset.")
            return self.data[idx], self.targets[idx]
        else:
            return self.data[idx]

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns
        -------
            int: The length of the dataset.

        Raises
        ------
            ValueError: If the data is not loaded.
        """
        if self.data is None:
            raise ValueError("The data is not loaded. Please load the data before using the dataset.")

        return len(self.data)

    def _prepare(self) -> None:
        """Prepare the dataset by downloading and extracting it."""
        # replace with download, but not all datasets are downloadable with the method
        load_classification(name=self.name, split=self.split, extract_path=self.save_path)

    def _load(self) -> None:
        """Load the dataset and fit the self.transform object to the data and targets."""
        data: np.ndarray
        targets: np.ndarray
        data, targets = load_classification(name=self.name, split=self.split, extract_path=self.save_path)

        transform: bt.Compose = bt.Compose([ToTorchTensor(), LabelTransform()])
        transform.fit(data, targets)

        self.data, self.targets = transform(data, targets)

        self.transforms.fit(self.data, self.targets)
