"""The different transforms in the torchchronos library.

All transformations inherit from the transform class. The base Transform class provides basic features like
__call__(), fit() or transform(). The base class is implemented in a way, that each transform has a fit()
and a transform() method. This method make basic checks and possibly reshape the input. The _real_
transformations are happening in the inherited _fit() and _transform() methods.

Each new implemented transform only has to implement the _fit(), _transform(), _inverse() and __repr__()
methods. The rest is already done in the base class. To implement new transforms it can be assumed that the
time series paramter is a tensor with 3 dimenstins: (n_timeseries, n_dimensions, n_timesteps). A transform
with less dimensions is first reshaped into the correct size, to ensure this property.

There are 3 possible verions of data for the transforms: 1, 2, and 3 dimensional tensors. A 3 dimensional
tensor is just passed to the transform. For a 2 dimenstional tensor it is assumed that the dimesions are
(n_timeseries, n_time_steps) and a syntetic n_dim 1 is added. For 1 dimenstional data also a n_timeseries
is added. This reshape is automaticly reversed in the transform method of the base class, and thereofore
does not impact the user.

"""

from .base_transforms import Compose, Transform
from .basic_transforms import Identity, NaNToNumber, Normalize, Scale, Shift
from .format_conversion_transforms import To, ToNumpyArray, ToTorchTensor
from .representation_transformations import (
    CombineToComplex,
    ComplexToPolar,
    LabelTransform,
    PolarToComplex,
    SplitComplexToRealImag,
)
from .structure_transforms import Crop, Filter, PadBack, PadFront, SlidingWindow
from .transformation_exceptions import NoInverseError

__all__ = [
    "Compose",
    "Transform",
    "LabelTransform",
    "PolarToComplex",
    "ComplexToPolar",
    "CombineToComplex",
    "SplitComplexToRealImag",
    "ToTorchTensor",
    "ToNumpyArray",
    "To",
    "Normalize",
    "Identity",
    "Scale",
    "Shift",
    "Filter",
    "Crop",
    "PadBack",
    "PadFront",
    "SlidingWindow",
    "NoInverseError",
    "NaNToNumber",
]
