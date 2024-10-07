from pathlib import Path
import tempfile
import urllib.request
import os

from torchchronos.datasets.ts_pile_paths import ts_pile_all
from torchchronos.datasets.prepareable_dataset import PrepareableDataset
from torchchronos.transforms.base_transforms import Transform
from torchchronos.transforms.basic_transforms import Identity
from aeon.datasets._data_loaders import load_from_tsfile, load_from_tsf_file
from torchchronos import dataset_cache_path
import numpy as np
import torch



class TSPileDataset(PrepareableDataset):
    def __init__(self,
        name: str,
        split: str | None = None,
        path: Path | str | None = os.path.join(dataset_cache_path, "ts_pile"),
        return_labels: bool = True,
        transform: Transform = Identity()):
        self.name = name
        self._data = None
        self._targets = None
        if isinstance(path, str):
            self._save_path = Path(path)
        elif isinstance(path, Path):
            self._save_path = path
        self._information_dict = None
        self._has_target = False
        
        super().__init__(transform=transform)


    def _prepare(self):
        # create save path
        os.makedirs(self._save_path, exist_ok=True)

        # check if name is in ts_pile_all
        for ts_pile_dict in ts_pile_all:
            if self.name in ts_pile_dict["dataset_names"]:
                self._information_dict = ts_pile_dict
                break
        
        if self._information_dict is None:
            raise ValueError(f"Dataset {self.name} not found in ts_pile_all")
        
        # check if data is already downloaded
        self._save_path = os.path.join(self._save_path, f"{self.name}.npz")
        if os.path.exists(self._save_path):
            if self._information_dict["format"] == "ucr":
                self._has_target = True
            elif self._information_dict["format"] == "tsf":
                self._has_target = False
            return
        
        # download data
        with tempfile.TemporaryDirectory() as tmpdirname:
            format = self._information_dict["format"]
            if format == "ucr":
                for split in ["TRAIN", "TEST"]:
                    url = self._information_dict["base_url"] + f"{self.name}/" +self.name + f"_{split}.ts"
                    urllib.request.urlretrieve(url, tmpdirname + f"{self.name}_{split}.ts")
                train = load_from_tsfile(tmpdirname + f"{self.name}_TRAIN.ts")
                test = load_from_tsfile(tmpdirname + f"{self.name}_TEST.ts")
                data = np.concatenate((train[0], test[0]), axis=0)
                targets = np.concatenate((train[1], test[1]), axis=0)
                self._has_target = True
            elif format == "tsf":
                url = self._information_dict["base_url"] + f"{self.name}.tsf"
                urllib.request.urlretrieve(url, tmpdirname + f"{self.name}.tsf")
                df, meta_data = load_from_tsf_file(tmpdirname + f"{self.name}.tsf")
                if meta_data["contain_equal_length"] == True:
                    data = df["series_value"]
                    data = data.to_numpy()
                else:
                    df["len"] = df["series_value"].apply(len)
                    max_len = len(df.sort_values(['len'],ascending=False).groupby(['len']).transform(min).head(1)["series_value"].values[0])
                    padded_data = []
                    for i in range(len(df)):
                        padded_data.append(np.pad(df["series_value"][i], (0, max_len - len(df["series_value"][i])), 'constant', constant_values=(np.nan)))
                    data = np.stack(padded_data)
                self._has_target = False
            elif format == "csv":
                url = self._information_dict["base_url"] + f"{self.name}.csv"
                print(url)
                urllib.request.urlretrieve(url, tmpdirname + f"{self.name}.csv")
                data = np.genfromtxt(tmpdirname + f"{self.name}.csv", delimiter=",")
                self._has_target = False
                

        
        with open(self._save_path, "wb") as f:
            if self._has_target:
                np.savez(f, data=data, targets=targets)
            else:
                np.savez(f, data=data)

    def _load(self):
        data = np.load(self._save_path)
        self._data = torch.from_numpy(data["data"])
        if self._has_target:
            targets = np.load(self._save_path)["targets"]

            self._targets = torch.from_numpy(targets.astype(float))

    def _get_item(self, index):
        if self._has_target:
            return self._data[index], self._targets[index]
        else:
            return self._data[index]

    def __len__(self):
        return len(self._data)
    


if __name__ == "__main__":
    for dataset in ["national_illness", "traffic", "weather"]:
        dataset = TSPileDataset(dataset)
        dataset.prepare()
        dataset.load()
        print(dataset[0].shape)