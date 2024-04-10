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
wafer = AeonClassificationDataset("Wafer")
ddm = DatasetDataModule(gun_point, test=wafer)
ddm.prepare_data()
ddm.setup("fit")
ddm.setup("test")

train_loader = ddm.train_dataloader()

for batch in train_loader:
    data, targets = batch
    print(data.shape)

print(len(ddm.train_dataset))
print(len(ddm.test_dataset))
