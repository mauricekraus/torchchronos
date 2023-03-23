# torchchronos
**torchchronos* is an experimental PyTorch and Lightning compatible library that provides easy and flexible access to various time-series datasets for classification and regression tasks. It also provides a simple and extensible transform API to preprocess data.
It is inspired by the much more complex [torchtime](https://github.com/philipdarke/torchtime).

## Installation
You can install torchchronos via pip:

`pip install torchchronos`

## Usage
### Datasets
torchchronos currently provides access to several popular time-series datasets, including:

- UCR/UEA Time Series Classification Repository

To use a dataset, you can simply import the corresponding dataset class and create an instance:

```python
from torchchronos.datasets import UCRUEADataset
from torchchronos.transforms import PadFront

dataset = UCRUEADataset('ECG5000',path=Path(".cache/data"), transforms=PadFront(10))
```

### Data Modules
torchchronos also provides a multi gpu Lightning compatible DataModules to make it easy to load and preprocess data. For example:

```python
from torchchronos.lightning import UCRUEAModule
from torchchronos.transforms import PadFront

module = UCRUEAModule('ECG5000', split_ratio= (0.75, 0.15), batch_size= 32) transforms=Compose([PadFront(10), PadBack(10)]))
```

### Transforms
torchchronos provides a flexible transform API to preprocess time-series data. For example, to normalize a dataset, you can define a transform like this:

```python
from torchchronos.transforms import Transform

class Normalize(Transform):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        return (data - self.mean) / self.std
```

## Roadmap
The following features are planned for future releases of torchchronos:

Support for additional time-series datasets, including:
- Energy consumption dataset
- Traffic dataset
- PhysioNet Challenge 2012 (in-hospital mortality)
- PhysioNet Challenge 2019 (sepsis prediction) datasets
Additional transform classes, including:
- Resampling
- Missing value imputation

If you have any feature requests or suggestions, please open an issue on our GitHub page.