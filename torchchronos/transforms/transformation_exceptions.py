"""Collection of some used exceptions in the package."""


class NoInverseError(Exception):
    """Raised when the inverse of a transformation is not possible."""

    pass


class TimeSeriesNotLongEnoughError(Exception):
    """Raised when the time series is not long enough for the transformation."""

    pass
