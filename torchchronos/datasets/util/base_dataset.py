from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data, targets=None):
        self.data = data
        self.targets = targets

    def __getitem__(self, idx):
        if self.targets is None:
            return self.data[idx], None
        else:
            return self.data[idx], self.targets[idx]
        
