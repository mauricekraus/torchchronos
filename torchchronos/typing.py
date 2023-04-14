from os import PathLike
from typing import Literal, TypeAlias, Union

AnyPath: TypeAlias = Union[str, bytes, PathLike]
DatasetSplit: TypeAlias = Union[Literal["train"], Literal["val"], Literal["test"]]
