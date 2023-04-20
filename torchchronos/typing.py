from os import PathLike
from typing import TypeAlias
from enum import Enum

AnyPath: TypeAlias = str | bytes | PathLike


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
