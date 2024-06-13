import torch


def dataset_1():
    torch.torch.manual_seed(7)
    time_series = torch.randint(0, 100, (100, 1, 52)).type(torch.float32)
    return time_series


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


datasets = [dataset_1(), dataset_2()]
targets = [targets_none, targets_random]


test_cases = []

for dataset in datasets:
    for target_method in targets:
        size = dataset.shape[0]
        target = target_method(size)
        test_cases.append((dataset, target, False))
        test_cases.append((make_time_series_unequal_length(dataset), target, True))
