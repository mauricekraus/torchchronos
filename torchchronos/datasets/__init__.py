"""This module is for loading different Datasets.

There are 3 main ways to gain data to perform experiements on. The first is loading a dataset with the AeonClassificatin or MonashForcasting classes.
These classes use the Aeon library to download the data that is then prepared and stored.

The second possibility is to download a numpy file and use the CachedDataset class to load the data. 

The last ways is to combine different datasets into a new one. This can be done with the ConcatDataset class. This is mainly used for mixing different
datatypes or creating a dataset with different time series data.

Note that all dataset classes have a prepare() and a load() method, that have to be called before using the data."""

from .aeon_datasets import AeonClassificationDataset, MonashForcastingDataset
from .cached_datasets import CachedDataset
from .concat_dataset import ConcatDataset, FrequencyMode, ShuffleMode
from .prepareable_dataset import NotLoadedError, NotPreparedError, PrepareableDataset

__all__ = [
    "AeonClassificationDataset",
    "MonashForcastingDataset",
    "CachedDataset",
    "PrepareableDataset",
    "ConcatDataset",
    "FrequencyMode",
    "ShuffleMode",
    "NotPreparedError",
    "NotLoadedError",
]
