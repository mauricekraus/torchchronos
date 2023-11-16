import torch
from torch.utils.data import Dataset
from typing import Any

from aeon.datasets._data_loaders import load_classification, load_forecasting, load_regression
from aeon.datasets import tsc_data_lists, tser_data_lists, tsf_data_lists


def load_dataset(dataset_name: str, split = None) -> Any:
    """Load a dataset from the Monash tser data archives.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str, optional): ["train", "test", None] The split to load. Defaults to None. If None, the entire dataset is loaded.
    Returns:
        Any: The dataset.
    """

    assert split in ["train", "test", None], ValueError("Split must be one of 'train', 'test' or None.")

    if dataset_name in tsc_data_lists.univariate or dataset_name in tsc_data_lists.multivariate:
        return ClassificationDataset(dataset_name, split=split)
    elif dataset_name in tser_data_lists.tser_all:
        return RegressionDataset(dataset_name,  split=split)
    elif dataset_name in tsf_data_lists.tsf_all:
        raise ValueError("Not implemented yet.")
        return ForcastingDataset(dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not found in the Monash tser data archives.")
    

class ClassificationDataset(Dataset):

    def __init__(self, name : str, split = None):
        super().__init__()
        self.X, self.y, self.meta_data = load_classification(name, split = split, extract_path="data/")
        self.X = self.X.astype('float32')

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class RegressionDataset(Dataset):

    def __init__(self, name : str, split = None):
        super().__init__()
        self.X, self.y, self.meta_data = load_regression(name, split = split, extract_path="data/")
        
        self.X = self.X.astype('float32')

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class ForcastingDataset(Dataset):

    def __init__(self, name : str):
        super().__init__()
        self.X, self.meta_data = load_forecasting(name, extract_path="data/")
        print(type(self.X))
        self.X = self.X.astype('float32')
        

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]


    
