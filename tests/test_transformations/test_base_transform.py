from torchchronos.transforms.base_transforms import Transform, Compose
from torchchronos.transforms.format_conversion_transforms import ToTorchTensor
from torchchronos.transforms.structure_transforms import SlidingWindow, RemoveLabels
from torchchronos.transforms.transforms import Shift
from torchchronos.datasets.util.aeon_datasets import AeonClassificationDataset


def test_transform():
    pass


def test_compose():
    dataset = AeonClassificationDataset(name="GunPoint")
    dataset.prepare()
    dataset.load()

    transform = Compose([ToTorchTensor(), SlidingWindow(10, 3), Shift(shift=1)])
    transform.fit(dataset.data, dataset.targets)
    transformed_dataset = transform(dataset)

    dataset = AeonClassificationDataset(name="GunPoint")
    dataset.prepare()
    dataset.load()

    transform = Compose([ToTorchTensor(), RemoveLabels(), SlidingWindow(10, 3), Shift(shift=1)])
    transform.fit(dataset.data, dataset.targets)
    transformed_dataset = transform(dataset)
