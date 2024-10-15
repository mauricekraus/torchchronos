"""Collection of util functions for datasets."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


def save_dataset(dataset: TensorDataset, name: str, save_path: Path | None = None) -> None:
    """Save a dataset to a file.

    Args:
        dataset: The dataset to save.
        name: The name of the file.
        save_path: The path to save the file to.

    """
    has_targets: bool = True if isinstance(dataset[0], tuple) else False
    data: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for i in range(len(dataset)):
        if has_targets:
            ts, tar = dataset[i]
            data.append(ts)
            targets.append(tar)
        else:
            data.append(dataset[i])

    data = torch.cat(data).numpy()
    if has_targets:
        targets = torch.stack(targets).numpy()

    if save_path is None:
        save_path = Path(".cache/torchchronos/datasets")

    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / f"{name}.npz", "wb") as f:
        if has_targets:
            np.savez(f, data=data, targets=targets)
        else:
            np.savez(f, data=data)
