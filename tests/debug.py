from torchchronos.datasets.aeon_datasets import AeonClassificationDataset

from torchchronos.lightning.dataset_data_module import DatasetDataModule


dataset = AeonClassificationDataset("GunPoint")
dataset.prepare()
dataset.load()
print(dataset[0][0].shape)
dm = DatasetDataModule(dataset, dataset, dataset)
dm.prepare_data()

dm.setup("fit")

data_loader = dm.train_dataloader()

for batch in data_loader:
    print(batch[0].shape)
