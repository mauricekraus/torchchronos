from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..download import download_and_unzip_dataset
from ..transforms import Transform
from ..typing import DatasetSplit, AnyPath


class TFCPretrainDataset(Dataset):
    """Loads certain pieces of data from the TF-C paper [1]_ [2]_.

    Supported datasets:
        - "Gesture"
        - "EMG"

    They are preprocessed in the exact same way as in the original paper.

    The data will be downloaded to the directory specified by ``path`` if not already
    present. The data will be stored in a subdirectory named after the dataset.

    Args:
        name: Name of the dataset to load (see above for supported datasets).
        path: Path to the directory where the dataset should be stored.
        split: Which split of the dataset to load. Can be "train", "val" or "test".
        transform: A optional :class:`~Transform` to apply to the data.

    Shapes:
        The returned data is a tuple containing the data and the label. The data is a
        :class:`torch.FloatTensor` of shape ``(batch, sequence length, number of features)`` and the label is a
        :class:`torch.LongTensor` of shape ``(batch,)``.

    References:
        .. [1]
            Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency
            Xiang Zhang, Ziyuan Zhao, Theodoros Tsiligkaridis, and Marinka Zitnik
            Proceedings of Neural Information Processing Systems, NeurIPS 2022
            https://arxiv.org/abs/2206.08496.
        .. [2]
            Overview over the originally uploaded datasets on Figshare:
            https://figshare.com/search?q=%22Self-Supervised%20Contrastive%20Pre-Training%20For%20Time%20Series%20via%20Time-Frequency%20Consistency%22
    """

    NAME_TO_URL = {
        "Gesture": "https://figshare.com/ndownloader/articles/22634401/versions/2",
        "EMG": "https://figshare.com/ndownloader/articles/22634332/versions/2",
    }

    def __init__(
        self,
        name: str,
        path: AnyPath,
        split: DatasetSplit = DatasetSplit.TRAIN,
        transform: Transform | None = None,
    ) -> None:
        super().__init__()

        self.name = name
        self.path = path
        self.split = split
        self.transform = transform

        download_tfc_pretrain(self.path, self.name)
        self.full_path = Path(path) / name / f"{split.value}.pt"

        data = torch.load(self.full_path)
        self._samples = data["samples"].transpose(1, 2)  # move features to last dim
        self._labels = data["labels"].long()
        assert len(self._samples) == len(
            self._labels
        ), "Samples and labels must have the same length"

        if self.transform is not None:
            self.transform = self.transform.fit(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.LongTensor]:
        x = self._samples[index, ...]
        if self.transform is not None:
            x = self.transform(x)
        return x, self._labels[index]


def download_tfc_pretrain(path: AnyPath, name: str) -> None:
    dataset_path = Path(path) / name

    if not dataset_path.exists():
        url = TFCPretrainDataset.NAME_TO_URL[name]
        # This downloads all splits at once
        download_and_unzip_dataset(url, dataset_path)

    if not dataset_path.exists():
        raise RuntimeError(
            f"Downloaded data to {dataset_path.absolute()} but it does not exist."
        )
