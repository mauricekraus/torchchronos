class MissingValueError(Exception):
    """Indicates that a time series has at least one missing value."""
    def __init__(self, msg):
        super().__init__(msg)
