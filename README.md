# Torchchronos
## Content
The goal of this project is to make loading time series and applying transformations in PyTorch easier. The library is divided into three main components: datasets, lightning and transforms. 

The datasets section provides some methods for loading online datasets. Datasets that can be loaded are:
+ every dataset from https://timeseriesclassification.com/
+ every dataset form https://forecastingdata.org/

Currently all datasets are downloaded using the [Aeon Library](https://www.aeon-toolkit.org/en/stable/).
The datasets module also provides a `PrepareableDataset` class, introducing a two step method to handle data. First the data is prepared, often the dataset files are downlowded here, and second the data is loaded. This is for splitting downloading the data and loading them into memory. Many torchchronos dataset classes inherit from the `PrepareableDataset` class.

The lightning section provides a lightning moduls for a prepareable dataset. The data is setup and loaded in the respectfully methods and all datas handeling is hidden in the class. In combination with the `ConcatDataset`class, the lightning module allows to train and test on different datasets.

The transformation section provides different methods to transform the datasets. As with other transformation classes, the transformations have to be fit to the data to later transform them.

```python
from torchchronos.datasets import AeonClassificationDataset
from torchchronos.transforms import Compose, Normalize, PadFront

time_series = AeonClassificationDataset("GunPoint")
time_series.prepare()
time_series.load()

transform = Compose([Normalize(), PadFront(10)])
transform.fit(time_series)

transformed_time_series = transform(time_series)

```

## Installation

The library can be installed via pip
```bash
pip intall torchchronos
```
