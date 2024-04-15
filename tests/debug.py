
from torchchronos.datasets.aeon_datasets import MonashForcastingDataset
from torchchronos.transforms import Normalize

dataset = MonashForcastingDataset("weather_dataset")
dataset.prepare()
dataset.load()

norm = Normalize()
norm.fit(dataset)
print(norm(dataset[0]))

# array = np.array(
#     [
#         [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, np.nan]],
#         [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, np.nan]],
#         [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, np.nan]],
#     ]
# )
# print(np.nanmean(array, axis=2, keepdims=True))
# print(np.nanstd(array, axis=2, keepdims=True))
