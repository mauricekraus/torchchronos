import torch

from .base import Transform


class PadFront(Transform):
    def __init__(self, length):
        self.length = length

    # Not needed in Padding
    def fit(self, ts):
        pass

    def transform(self, ts):
        return torch.cat([torch.zeros(self.length, ts.shape[1]), ts], dim=0)

    def __repr__(self):
        return self.__class__.__name__ + "(length={0})".format(self.length)


class PadBack(Transform):
    def __init__(self, length):
        self.length = length

    def transfrom(self, ts):
        return torch.cat([ts, torch.zeros(self.length, ts.shape[1])], dim=0)

    def __repr__(self):
        return self.__class__.__name__ + "(length={0})".format(self.length)
