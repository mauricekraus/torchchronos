"""Class for using the UCR datasets from the Aeon library."""

import tempfile
from pathlib import Path

import numpy as np
import torch
from aeon.datasets._data_loaders import load_classification, load_forecasting

from torchchronos import dataset_cache_path

from torchchronos.transforms.base_transforms import Transform
from torchchronos.transforms.representation_transformations import LabelTransform
from torchchronos.transforms.basic_transforms import Identity
from torchchronos.datasets.prepareable_dataset import NotLoadedError, PrepareableDataset


class AeonClassificationDataset(PrepareableDataset):
    """A Dataset class to load classification datasets from the Aeon library (UCR).

    This class is a PrepareableDataset and therefore has a prepare and a load step. In the prepare step,
    the dataset is downloaded and extracted. In the load step, the data is loaded into the memory and
    transformed. Noth prepare and load have to be called before the dataset can be used.

    Args:
        name: The name of the dataset.
        split: The split of the dataset.
        path: The path to save the dataset.
        return_labels: Whether to return labels along with the data. Defaults to True.
        transform: The data transformation to apply. Defaults to Identity().

    Examples:
        Load the GunPoint dataset and get the first item.

        >>> dataset = AeonClassificationDataset(name="GunPoint")
        >>> dataset.prepare()
        >>> dataset.load()
        >>> data, label = dataset[0]

        Loat the train split, apply a scaling transformation and only return the data without the respective
        labels.

        >>> from torchchronos.transforms import Scale
        >>> scale_transform = Scale(10)
        >>> dataset = AeonClassificationDataset(
        ...     name="GunPoint", split="train", transform=scale_transform, return_labels=False
        ... )
        >>> dataset.prepare()
        >>> dataset.load()
        >>> data = dataset[0]  # Scaled by 10

    """

    def __init__(
        self,
        name: str,
        split: str | None = None,
        path: Path | str | None = None,
        return_labels: bool = True,
        transform: Transform = Identity(),
    ) -> None:
        self._data: torch.Tensor | None = None
        self._targets: torch.Tensor | None = None

        self.name: str = name
        self.split: str | None = split
        self._save_path: Path | None = None
        if path is None:
            self._save_path = None
        elif isinstance(path, str):
            self._save_path = Path(path)
        elif isinstance(path, Path):
            self._save_path = path
        else:
            raise TypeError("The 'path' argument must be of type 'str' or 'Path'.")

        self.return_labels: bool = return_labels

        super().__init__(
            transform=transform,
        )

    def _get_item(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            The item from the dataset. If `return_labels` is True, a tuple of the data and the target is
            returned.
            If not the data is returned, however not in a tuple.

        Raises:
            ValueError: If the data is not loaded or the targets are not loaded.
        """
        if self._data is None:
            raise ValueError("The data is not loaded. Please load the data before using the dataset.")

        if self.return_labels is True:
            if self._targets is None:
                raise ValueError("The targets is not loaded. Please load the data before using the dataset.")
            return self._data[idx], self._targets[idx]
        else:
            return self._data[idx]

    def __len__(self) -> int:
        if self._data is None:
            raise ValueError("The data is not loaded. Please load the data before using the dataset.")

        return len(self._data)

    def _prepare(self) -> None:
        """Prepare the dataset by downloading and extracting it."""
        load_classification(name=self.name, split=self.split, extract_path=self._save_path)

    def _load(self) -> None:
        """Load the dataset and fit the self.transform object to the data and targets.

        First the already downloaded data is loaded into memory. Then the data is converted to a torch.Tensor.
        With the `LabelTransform` the targets are converted into integer targets from [0, n_classes - 1].
        Lastly the transforms being applied each time a item is retrieved are fitted to the data and targets.
        """
        data: np.ndarray
        targets: np.ndarray
        data, targets = load_classification(name=self.name, split=self.split, extract_path=self._save_path)

        data = torch.tensor(data)
        targets = targets.astype(float)

        targets = torch.tensor(targets)
        transform: LabelTransform = LabelTransform()
        transform.fit(data, targets)

        self._data, self._targets = transform(data, targets)

        self.transforms.fit(self._data, self._targets)


class MonashForcastingDataset(PrepareableDataset):
    """Dataset class for loading datasets from the Monash Repository.

    The dataset that can be downloaded can be checked here: https://forecastingdata.org/
    However, the datasets are downloaded via the aeon library and therefore the names are slightly different.
    The dataset names can be checked here:
    https://github.com/aeon-toolkit/aeon/blob/main/aeon/datasets/tsf_datasets.py

    Since many of the time series have different lengths, the time series are padded with np.nan to the
    maximum length. This is a PrepareableDataset, in the prepare step the dataset is downloaded, extracted,
    padded and saved as a .npy file. In the load step, the data is loaded into memory and transformed.

    Args:
        name: The name of the dataset.
        path: The path to save the dataset.
        transform: The data transformation to apply.

    Examples:
        Load the dataset and transform it.

        >>> from torchchronos.transforms import Scale
        >>>
        >>> scale_transform = Scale(10)
        >>> dataset = MonashForcastingDataset(
        ...     name="sunspot_dataset_without_missing_values", transform=scale_transform
        ... )
        >>> dataset.prepare()
        >>> dataset.load()
        >>> data = dataset[0]
    """

    def __init__(self, name: str, path: Path | str = dataset_cache_path, transform:Transform=Identity()):
        self._data: torch.Tensor | None = None

        self.name: str = name
        if isinstance(path, str):
            self._save_path = Path(path)
        elif isinstance(path, Path):
            self._save_path = path

        super().__init__(
            transform=transform,
        )

    def _get_item(self, idx: int) -> torch.Tensor:
        """Get an item from the dataset."""
        if self._data is None:
            raise NotLoadedError("The Dataset has to be loaded fitst, before an item can be indexed.")
        return self._data[idx]

    def __len__(self) -> int:
        if self._data is None:
            raise NotLoadedError("The Dataset has to be loaded fitst, before the length can be accessed.")
        return len(self._data)

    def _prepare(self) -> None:
        """Prepare the dataset by downloading and padding np.nan's to it.

        This Method consists of 3 main steps:
        1. Download the dataset from the Monash Repository. This is done via the Aeon libarary.
        2. Pad the time series with np.nan's to the maximum length.
        3. Save the padded time series as a .npy file.
        """
        if (self._save_path / f"{self.name}.npy").exists():
            return
        with tempfile.TemporaryDirectory() as tmpdirname:
            df = load_forecasting(name=self.name, extract_path=tmpdirname)
            data = []
            max_len = max([len(df["series_value"][i]) for i in range(len(df))])
            for i in range(len(df)):
                time_series = df["series_value"][i].to_numpy()
                padded_time_series = np.pad(
                    time_series, (0, max_len - len(time_series)), "constant", constant_values=(np.nan,)
                )
                data.append(padded_time_series)

            data = np.array(data)
            with open(self._save_path / f"{self.name}.npy", "wb") as f:
                np.save(f, data)

    def _load(self) -> None:
        """Load the dataset and fit the self.transform object to the data."""
        np_data = np.load(self._save_path / f"{self.name}.npy")

        self._data = torch.tensor(np_data)
        self.transforms.fit(self._data)
