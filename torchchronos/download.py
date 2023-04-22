import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from sktime.datasets._data_io import _list_available_datasets
from .typing import AnyPath


from pathlib import Path
import zipfile


def _list_available_datasets(data_dir: AnyPath) -> list[str]:
    datasets = []
    for name in os.listdir(data_dir):
        sub_dir = os.path.join(data_dir, name)
        if os.path.isdir(sub_dir):
            all_files = os.listdir(sub_dir)
            if name + "_TRAIN.ts" in all_files and name + "_TEST.ts" in all_files:
                datasets.append(name)
    return datasets


def download_uea_ucr(dataset_name: str, extract_path: str | Path | None = None) -> bool:
    """
    Downloads the UEA/UCR time series dataset from sktime to the specified path.

    Parameters:
    dataset_name (str): The name of the dataset to be downloaded.
    extract_path (str | Path | None, default=None) : The path to extract the dataset. If None, extract to `.cache/data` directory.

    Returns:
    bool: True if the dataset was already present, False otherwise.

    Raises:
    ValueError: If the dataset is not available on the given path or on 'https://timeseriesclassification.com/'.
    """
    # Download UCR/UEA archive from sktime
    if extract_path is not None:
        local_dirname = Path(extract_path)
    else:
        local_dirname = Path(".cache/data")

    full_path = local_dirname / dataset_name

    full_path.mkdir(parents=True, exist_ok=True)

    if dataset_name not in _list_available_datasets(local_dirname):
        # Dataset is not already present in the datasets directory provided.
        # If it is not there, download and install it.
        url = f"https://timeseriesclassification.com/Downloads/{dataset_name}.zip"
        try:
            download_and_unzip_dataset(url, full_path)
            # If zip contains folder name as root, move contents up one level
            if len(os.listdir(full_path)) == 1:
                subfolder = os.listdir(full_path)[0]
                subfolder_path = full_path / subfolder
                for file in os.listdir(subfolder_path):
                    shutil.move(subfolder_path / file, full_path)
                os.rmdir(subfolder_path)
        except zipfile.BadZipFile as e:
            raise ValueError(
                f"Invalid dataset name '{dataset_name}' is not available on extract path '{extract_path}'. "
                f"Nor is it available on 'https://timeseriesclassification.com/'."
            ) from e
        return False
    else:
        return True


def download_and_unzip_dataset(url: str, path: AnyPath) -> None:
    with tempfile.NamedTemporaryFile() as zipped_file:
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, zipped_file)

        with zipfile.ZipFile(zipped_file) as zf:
            zf.extractall(path)
