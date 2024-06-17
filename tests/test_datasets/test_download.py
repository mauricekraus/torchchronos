import pytest
from aeon.datasets._data_loaders import load_forecasting, load_regression
from aeon.datasets.tser_data_lists import tser_all
from aeon.datasets.tsf_data_lists import tsf_all


@pytest.mark.download_all
@pytest.mark.parametrize("dataset_name", tsf_all)
def test_download_all_tsf(dataset_name, tmp_path):
    data = load_forecasting(dataset_name, tmp_path)
    len(data)


@pytest.mark.download_all
@pytest.mark.parametrize("dataset_name", tser_all)
def test_download_all_tser(dataset_name, tmp_path):
    data = load_regression(name=dataset_name, extract_path=tmp_path)
    len(data)
