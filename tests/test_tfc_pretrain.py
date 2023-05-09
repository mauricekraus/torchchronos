from torchchronos.typing import DatasetSplit
from torchchronos.datasets import TFCPretrainDataset
from torchchronos.lightning import TFCPretrainDataModule
from torchchronos.transforms import PadFront


def test_shapes_Gesture_train(tmp_path) -> None:
    """Test that the dataset returns the correct shapes."""

    dataset = TFCPretrainDataset(
        name="Gesture", path=tmp_path, split=DatasetSplit.TRAIN
    )
    assert len(dataset) == 320
    data, label = dataset[0]
    assert data.shape == (206, 3)
    assert label.shape == ()


def test_shapes_EMG_test(tmp_path) -> None:
    """Test that the dataset returns the correct shapes."""

    dataset = TFCPretrainDataset(name="EMG", path=tmp_path, split=DatasetSplit.TEST)
    assert len(dataset) == 41
    data, label = dataset[1]
    assert data.shape == (1_500, 1)
    assert label.shape == ()

def test_shapes_ECG_test(tmp_path) -> None:
    """Test that the dataset returns the correct shapes."""

    dataset = TFCPretrainDataset(name="ECG", path=tmp_path, split=DatasetSplit.TEST)
    assert len(dataset) == 43_673
    data, label = dataset[1]
    assert data.shape == (1_500, 1)
    assert label.shape == ()


def test_datamodule() -> None:
    batch_size = 47
    data_module = TFCPretrainDataModule(name="EMG", batch_size=batch_size)
    data_module.prepare_data()
    data_module.setup("fit")
    batch_x, batch_y = next(iter(data_module.train_dataloader()))
    assert batch_x.shape[1] == batch_size
    assert batch_y.shape[0] == batch_size


def test_datamodule_transform() -> None:
    batch_size = 3
    transform = PadFront(99)
    data_module = TFCPretrainDataModule(
        name="EMG", batch_size=batch_size, transform=transform
    )
    data_module.prepare_data()
    data_module.setup("fit")
    batch_x, batch_y = next(iter(data_module.train_dataloader()))
    assert batch_x.shape == (1599, batch_size, 1)
    assert batch_y.shape == (batch_size,)
