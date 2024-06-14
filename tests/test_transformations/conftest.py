import pytest
import torch


# Dataset generation functions
def dataset_1():
    torch.manual_seed(7)
    return torch.randint(0, 100, (100, 1, 52)).type(torch.float32)


def dataset_2():
    return torch.arange(0, 200, 1).repeat(200).reshape(200, 1, 200).float() / 100 - 1


def targets_none(size):
    return None


def targets_random(size):
    return torch.randint(0, 10, (size, 1))


def make_time_series_unequal_length(ts):
    test_ts_unequal_length = ts.clone().detach()
    for i in range(ts.shape[0] - 1):
        rand_int = torch.randint(low=1, high=ts.shape[2] - 1, size=(1,)).item()
        test_ts_unequal_length[i, 0, rand_int:] = float("nan")
    return test_ts_unequal_length


# Register datasets and target functions in lists for easy extension
datasets = [dataset_1, dataset_2]
targets = [targets_none, targets_random]


# Fixtures for parameterization
@pytest.fixture(scope="module", params=datasets)
def dataset(request):
    return request.param()


@pytest.fixture(scope="module", params=targets)
def target_method(request):
    return request.param


@pytest.fixture(scope="module")
def targets(dataset, target_method):
    size = dataset.shape[0]
    return target_method(size)


@pytest.fixture(scope="module")
def unequal_length_dataset(dataset):
    return make_time_series_unequal_length(dataset)


@pytest.fixture(scope="module", params=[False, True])
def test_case_with_target(request, dataset, targets, unequal_length_dataset):
    if request.param:
        return unequal_length_dataset, targets, True
    else:
        return dataset, targets, False


@pytest.fixture(scope="module", params=[False, True])
def test_case_without_target(request, dataset, unequal_length_dataset):
    if request.param:
        return unequal_length_dataset, True
    else:
        return dataset, False
