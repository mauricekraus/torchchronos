from collections.abc import Sequence
import os
from pathlib import Path
from typing import Optional
import warnings
import zipfile

import numpy as np
import pandas as pd
from sktime.datasets import load_UCR_UEA_dataset
from sktime.datasets._data_io import _download_and_extract, _list_available_datasets




def _download_uea_ucr(extract_path: Optional[Path], dataset_name: str):
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

