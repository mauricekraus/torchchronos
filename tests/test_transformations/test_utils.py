import torch


def dataset_equal(d1, d2):
    has_targets = True if len(d1[0]) == 2 else False
    is_close = True
    for i in range(len(d1)):
        if not torch.allclose(d1[i][0], d2[i][0]):
            is_close = False
            break
        if has_targets:
            if not torch.allclose(d1[i][1], d2[i][1]):
                is_close = False
                break
    return is_close


def generate_dataset_1():
    torch.seed(7)
    time_series = torch.randn((100, 1, 52))
    return time_series


def generate_dataset_2():
    return torch.arange(0, 200, 1).repeat(200).reshape(200, 1, 200).float() / 100 - 1


ts_equal_length = [generate_dataset_1(), generate_dataset_2()]
