import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from sktime.datasets._data_io import _download_and_extract, _list_available_datasets
from .typing import AnyPath


def download_uea_ucr(extract_path: Optional[Path], dataset_name: str) -> None:
    """This downloads the uea ucr dataset from sktime to the path extract_path."""
    # Download UCR/UEA archive from sktime
    if extract_path is not None:
        local_dirname = extract_path
    else:
        local_dirname = Path(".cache/data")
    local_dirname.mkdir(parents=True, exist_ok=True)

    if dataset_name not in _list_available_datasets(local_dirname):
        # Dataset is not already present in the datasets directory provided.
        # If it is not there, download and install it.
        url = "https://timeseriesclassification.com/Downloads/%s.zip" % dataset_name
        # This also tests the validitiy of the URL, can't rely on the html
        # status code as it always returns 200
        try:
            _download_and_extract(
                url,
                extract_path=local_dirname,
            )
        except zipfile.BadZipFile as e:
            raise ValueError(
                f"Invalid dataset name ={dataset_name} is not available on extract path ="
                f"{extract_path}. Nor is it available on "
                f"https://timeseriesclassification.com/.",
            ) from e


def download_and_unzip_dataset(url: str, path: AnyPath) -> None:
    with tempfile.NamedTemporaryFile() as zipped_file:
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, zipped_file)

        with zipfile.ZipFile(zipped_file) as zf:
            zf.extractall(path)
