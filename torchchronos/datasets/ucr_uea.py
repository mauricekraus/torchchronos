from pathlib import Path
import pandas as pd
from sktime.datasets import load_UCR_UEA_dataset
from sktime.datasets._data_io import _load_provided_dataset
import torch
from torch.utils.data import Dataset
from ..typing import DatasetSplit
from ..transforms import Transform

from ..utils import parse_ts


class UCRUEADataset(Dataset):
    def __init__(
        self,
        ds_name: str,
        path: Path,
        split: DatasetSplit = DatasetSplit.TRAIN,
        transform: Transform | None = None,
    ) -> None:
        super().__init__()

        self.transform = transform
        if split == DatasetSplit.VAL:
            raise ValueError("UCR/UEA datasets do not have a validation split")
        split_val = "TRAIN" if split == DatasetSplit.TRAIN else "TEST"
        self.xs, self.ys = _load_provided_dataset(
            ds_name,
            split=split_val,
            return_type="numpy3d",
            local_module=path.parent,
            local_dirname=path.stem,
        )
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

        if (
            self.ys.dtype == "U2" or self.ys.dtype == "<U1" or self.ys.dtype == "<U3"
        ):  # todo need to capture the differnt U types
            # convert string labels to int
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
