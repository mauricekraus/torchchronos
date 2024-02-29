from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def save_dataset(dataset: Dataset, name: str, save_path: Path = None) -> None:
    has_targets = True if isinstance(dataset[0], tuple) else False
    data = dataset[0][0] if has_targets else dataset[0]
    targets = [dataset[0][1]] if has_targets else None
    print(dataset[0][0].shape)
    print(data.shape)
    for i in range(1, len(dataset)):
        if has_targets:
            data = torch.cat((data, dataset[i][0]), axis=0)
            targets.append(dataset[i][1])
        else:
            data = torch.cat((data, dataset[i]), axis=0)

    data = data.numpy()
    print(data.shape)
    if has_targets:
        targets = np.array(targets)

    if save_path is None:
        save_path = Path(".cache/torchchronos/datasets")

    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / f"{name}.npz", "wb") as f:
        if has_targets:
            np.savez(f, data=data, targets=targets)
        else:
            np.savez(f, data=data)


def get_meta_data(dataset: Dataset) -> dict:
    meta_data = {}
    meta_data["length"] = len(dataset)
    has_targets = True if isinstance(dataset[0], tuple) else False
    if has_targets:
        meta_data["data_shape"] = dataset[0][0].shape
        meta_data["target_shape"] = dataset[0][1].shape
    else:
        meta_data["data_shape"] = dataset[0].shape
    return meta_data
