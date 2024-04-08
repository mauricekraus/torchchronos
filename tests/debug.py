from torchchronos.datasets.aeon_datasets import AeonClassificationDataset, MonashForcastingDataset

from torchchronos.lightning.dataset_data_module import DatasetDataModule
from torchchronos.transforms import Compose, Shift
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from aeon.datasets._data_loaders import load_forecasting

from torchchronos.datasets import ConcatDataset, AeonClassificationDataset
from torchchronos.transforms import Scale, Crop
from torchchronos.datasets.concat_dataset import ShuffleMode, FrequencyMode


gun_point = AeonClassificationDataset("GunPoint")
gun_point.prepare()
gun_point.load()
wafer = AeonClassificationDataset("Wafer")
wafer.prepare()
wafer.load()

cd = ConcatDataset([gun_point, wafer], FrequencyMode.ALL_EQUAL, shuffle=ShuffleMode.ACROSS_DATASETS)

cd.prepare()
cd.load()
print(len(cd))
print(cd.frequency)

for i in range(10):
    print(cd[i][0].shape)
