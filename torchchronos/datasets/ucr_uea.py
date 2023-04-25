from pathlib import Path
import pandas as pd
from sktime.datasets._data_io import _load_provided_dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from ..errors import MissingValueError
from ..typing import DatasetSplit
from ..transforms import Transform
from ..utils import parse_ts


class UCRUEADataset(Dataset):
    """UCR/UEA dataset.

    Args:
        ds_name (str): Name of the dataset.
        path (Path): Path to the dataset storage.
        split (DatasetSplit, optional): Dataset split to use. Defaults to None.
        transform (Transform, optional): Transform to apply to the data. Defaults to None.
        torchchronos_cache (bool, optional): Whether to cache the data in torchchronos format. Defaults to True.
            This allows for much faster loading of the data once it has been loaded once.
            It is recommended to leave this as True unless space is an issue.
            The file is not compressed to allow for faster loading.
    """

    def __init__(
        self,
        ds_name: str,
        path: Path,
        split: DatasetSplit | None = None,
        transform: Transform | None = None,
        torchchronos_cache: bool = True,
        raise_on_missing: bool = True,
    ) -> None:
        super().__init__()

        self.transform = transform
        match split:
            case None:
                split_val = None
            case DatasetSplit.TRAIN:
                split_val = "TRAIN"
            case DatasetSplit.VAL:
                raise ValueError("UCR/UEA datasets do not have a validation split")
            case DatasetSplit.TEST:
                split_val = "TEST"

        tc_cache_dir = path / ds_name
        tc_cache_path = tc_cache_dir / f"{split_val}.npz"
        if torchchronos_cache and tc_cache_path.exists():
            # memory-mapping is slightly faster
            data = np.load(tc_cache_path, mmap_mode="r")
            self.xs, self.ys = data["xs"], data["ys"]
        else:
            self.xs, self.ys = _load_provided_dataset(
                ds_name,
                split=split_val,
                return_type="numpy3d",
                local_module=path.parent,
                local_dirname=path.stem,
            )
            if torchchronos_cache:
                tc_cache_dir.mkdir(parents=True, exist_ok=True)
                # Using savez instead of savez_compressed because the latter is much slower
                np.savez(tc_cache_path, xs=self.xs, ys=self.ys)

        self.xs = torch.tensor(self.xs, dtype=torch.float32).transpose(1, 2)
        if self.transform is not None:
            self.transform = self.transform.fit(self.xs)

        # It doesnt matter if test or train, but test is usually smaller
        ts_info = parse_ts(path / ds_name / f"{ds_name}_TEST.ts")
        self.dimensions = ts_info.dimensions
        self.num_classes = ts_info.num_classes
        self.series_length = ts_info.series_length
        self.equal_length = ts_info.equal_length
        self.univariate = ts_info.univariate
        self.missing = ts_info.missing

        if raise_on_missing and ts_info.missing:
            raise MissingValueError(
                f"Dataset {ds_name} contains NaN values. If this is intended behavior, set `raise_on_missing=False`"
            )

        # convert string labels to int
        if self.ys.dtype.kind in {"U", "S"}:
            factorized_y = pd.factorize(self.ys, sort=True)
            self.y_labels = factorized_y[1]
            self.ys = factorized_y[0]

        self.ys = torch.tensor(self.ys, dtype=torch.long)

    def label_from_index(self, index: int) -> str:
        return self.y_labels[int(index)]

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.LongTensor]:
        if self.transform:
            return self.transform(self.xs[index]), self.ys[index]
        return self.xs[index], self.ys[index]
