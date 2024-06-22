import pytest
from aeon.datasets._data_loaders import load_forecasting, load_regression, load_classification
from aeon.datasets.tser_datasets import tser_monash
from aeon.datasets.tsc_datasets import univariate, univariate2015, multivariate


@pytest.mark.download_all
@pytest.mark.parametrize("dataset_name", univariate)
def test_download_all_tsc(dataset_name, tmp_path):
    data = load_classification(name=dataset_name, extract_path=tmp_path)
    len(data)


@pytest.mark.download_all
@pytest.mark.parametrize("dataset_name", tser_monash)
def test_download_all_tser(dataset_name, tmp_path):
    data = load_regression(name=dataset_name, extract_path=tmp_path)
    len(data)
