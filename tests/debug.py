import numpy as np
import torch

# array = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(torch.float32).reshape(2, 1, 3)  # 0.75
# torch_std = torch.std(array, dim=(0), keepdim=True) + 1e-5
# np_std = np.nanstd(array.numpy(), axis=(0), keepdims=True, ddof=1) + 1e-5
# print(torch_std)
# print(np_std)

# time_series = torch.arange(10).repeat(10, 1, 1).float()  # 10 samples, 1 feature, 100 time points
# normalize_transform = Normalize(local=True)
# transformed_time_series = normalize_transform(time_series)  # Does not require fitting

# print(transformed_time_series[0])
# print(torch.arange(-1.4863, 1.4863, 0.2973))

test_list = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
print(test_list[0][0])

print(type(test_list[0][0]))
print(isinstance(np.ndarray, list))
print(isinstance(torch.Tensor, list))
test_tuple = (((1, 2, 3), (4, 5, 6)), ((7, 8, 9), (10, 11, 12)))
print(test_tuple[0][0])
