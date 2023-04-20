# torchchronos

[![PyPI version](https://img.shields.io/pypi/v/torchchronos.svg?color=blue)](https://pypi.org/project/torchchronos)
[![license](https://img.shields.io/pypi/l/torchchronos.svg?color=blue)](https://github.com/felixdivo/torchchronos/blob/main/LICENSE)
[![python version](https://img.shields.io/badge/python-3.10+-blue)](https://devguide.python.org/versions/)

[![test](https://github.com/mauricekraus/torchchronos/actions/workflows/main.yml/badge.svg)](https://github.com/mauricekraus/torchchronos/actions/workflows/main.yml)
[![code style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

*torchchronos* is an experimental [PyTorch](https://pytorch.org/) and [Lightning](https://lightning.ai/pytorch-lightning/) compatible library that provides easy and flexible access to various time-series datasets for classification and regression tasks. It also provides a simple and extensible transform API to preprocess data.
It is inspired by the much more complicated [torchtime](https://github.com/philipdarke/torchtime).

## Installation
You can install torchchronos via pip:

`pip install torchchronos`

## Usage
### Datasets
torchchronos currently provides access to several popular time-series datasets, including:

- [UCR/UEA Time Series Classification Repository](https://www.timeseriesclassification.com/): `torchchronos.datasets.UCRUEADataset`
- Time series as preprocessed in the [TFC paper](https://github.com/mims-harvard/TFC-pretraining): `torchchronos.datasets.TFCPretrainDataset` (datasets `Gesture` and `EMG`)

To use a dataset, you can simply import the corresponding dataset class and create an instance:

```python
from torchchronos.datasets import UCRUEADataset
from torchchronos.transforms import PadFront
from torchchronos.download import download_uea_ucr

download_uea_ucr(Path(".cache/data"), "ECG5000")
dataset = UCRUEADataset('ECG5000', path=Path(".cache") / "data", transforms=PadFront(10))
```

### Data Modules
torchchronos also provides [Lightning compatible `DataModules`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) to make it easy to load and preprocess data. They support common use cases like (multi-)GPU training and train/test/val-splitting out of the box. For example:

```python
from torchchronos.lightning import UCRUEADataModule
from torchchronos.transforms import PadFront, PadBack

module = UCRUEAModule('ECG5000', split_ratio= (0.75, 0.15), batch_size= 32,
                      transforms=Compose([PadFront(10), PadBack(10)]))
```

Analogous the the datasets above, these dataloaders are supported as of now, wrapping the respective datasets:
- `torchchronos.lightning.UCRUEADataModule`
- `torchchronos.lightning.TFCPretrainDataModule`

### Transforms
torchchronos provides a flexible transform API to preprocess time-series data. For example, to normalize a dataset, you can define a custom `Transform` like this:

```python
from torchchronos.transforms import Transform

class Normalize(Transform):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()

    def __call__(self, data):
        return (data - self.mean) / self.std
```

## Roadmap
The following features are planned for future releases of torchchronos:

- Support for additional time-series datasets, including:
    - Energy consumption dataset
    - Traffic dataset
    - PhysioNet Challenge 2012 (in-hospital mortality)
    - PhysioNet Challenge 2019 (sepsis prediction) datasets
- Additional transform classes, including:
    - Resampling
    - Missing value imputation

If you have any feature requests or suggestions, please open an issue on our GitHub page.
