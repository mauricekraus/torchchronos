from torchchronos.datasets import TFCPretrainDataset


def test_shapes_Gesture_train(tmp_path) -> None:
    """Test that the dataset returns the correct shapes."""

    dataset = TFCPretrainDataset(name="Gesture", path=tmp_path, split="train")
    assert len(dataset) == 320
    data, label = dataset[0]
    assert data.shape == (206, 3)
    assert label.shape == ()


def test_shapes_EMG_test(tmp_path) -> None:
    """Test that the dataset returns the correct shapes."""

    dataset = TFCPretrainDataset(name="EMG", path=tmp_path, split="test")
    assert len(dataset) == 41
    data, label = dataset[1]
    assert data.shape == (1_500, 1)
    assert label.shape == ()
