"""Module for transforms manipulating the structure of the underlying data."""

from collections.abc import Callable

import torch

from .base_transforms import Transform
from .transformation_exceptions import NoInverseError


class Crop(Transform):
    """Crop transformation that crops a given portion of the time series.

    Args:
        start: The starting index of the crop.
        end: The ending index of the crop.

    Examples:
    This example demonstrates how to crop the time series from index 2 to 8.

    >>> import torch
    >>>
    >>> data = time_series = torch.arange(10).repeat(10, 1, 1).float()
    >>> crop = Crop(2, 8)
    >>> crop.fit(data)
    >>> cropped_data = crop(data)
    >>> cropped_data.shape
    torch.Size([10, 1, 6])
    >>> cropped_data[0]
    tensor([[2., 3., 4., 5., 6., 7.]])
    """

    def __init__(self, start: int, end: int) -> None:
        super().__init__()
        self.start = start
        self.end = end

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Fit the crop transformation.

        Args:
            time_series: The input time series.
            targets : The target values associated with the time series.

        Raises:
            ValueError: If start or end is a negative integer, or if start is greater than or equal to end,
                        or if end is greater than the length of the time series.

        """
        if self.start < 0 or self.end < 0:
            raise ValueError("Start and end must be positive integers")
        if self.start >= self.end:
            raise ValueError("Start must be less than end")
        if self.end > time_series.shape[-1]:
            raise ValueError("End must be less than the length of the time series")

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the crop transformation to the input time series.

        Args:
            time_series: The input time series.
            targets: The target values associated with the time series.

        Returns:
            The cropped time series and the targets (unchanged).

        """
        return time_series[:, :, self.start : self.end], targets

    def _invert(self) -> Transform:
        """Invert the crop transformation.

        Raises:
            NoInverseError: If the crop transformation is not invertible.

        """
        error_message = "Croping can not be inverted."
        raise NoInverseError(error_message)

    def __repr__(self) -> str:
        return f"Crop(start={self.start}, end={self.end})"


class PadFront(Transform):
    """Class to pad the front of the time series with zeros.

    This class has 2 different modes. This first mode is for padding at the front a specific number of zeros.
    The second pads the time series to a specific length by calculating the difference between the length of
    the time series and the desired length.

    Args:
        length: The length of the padding to be added.
        fixed_length: If True, the length is the time series will have the given length, after transforming.
        value: The value the time series is padded with. Defaults to 0.

    Examples:
    This example demonstrates how to pad 2 zeros at the front of the time series.

    >>> import torch
    >>> time_series = time_series = torch.arange(10).repeat(10, 1, 1).float()
    >>> pad = PadFront(2)
    >>> pad.fit(time_series)
    >>> padded_data = pad(time_series)
    >>> padded_data.shape
    torch.Size([10, 1, 12])
    >>> padded_data[0]
    tensor([[0., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]])

    Here the time series should be padded to a lenght of 15, the transform calculates, how many zeros to pad.

    >>> import torch
    >>> time_series = torch.arange(10).repeat(10, 1, 1).float()
    >>> pad = PadFront(15, True)
    >>> pad.fit(time_series)
    >>> padded_data = pad(time_series)
    >>> padded_data.shape
    torch.Size([10, 1, 15])
    >>> padded_data[0]
    tensor([[0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]])
    """

    def __init__(
        self,
        length: int,
        fixed_length: bool = False,
        value: float = 0,
    ) -> None:
        super().__init__()
        self.length = length
        self.fixed_length = fixed_length
        self.time_series_length: int | None = None
        self.value = value

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Fit the transformation by determining the length of the time series.

        Args:
            time_series: The input time series.
            targets: The target values associated with the time series.


        """
        self.time_series_length = time_series.shape[-1]
        if self.fixed_length:
            if self.length < self.time_series_length:
                raise RuntimeError("Pad length is less than the time series length")
            self.length = self.length - self.time_series_length

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the transformation by padding the front of the time series with zeros.

        Args:
            time_series: The input time series.
            targets: The target values associated with the time series.


        Returns:
            The transformed time series and targets (if provided).
        """
        if self.time_series_length is None:
            raise Exception("Fit must be called before transforming")

        padding = torch.zeros((time_series.shape[0], time_series.shape[1], self.length)) + self.value
        return torch.cat([padding, time_series], dim=2), targets

    def _invert(self) -> Transform:
        """Invert the transformation by cropping the padded front.

        Returns:
            The inverted transformation.

        Raises:
            Exception: If the fit method has not been called before inverting.
        """
        if self.time_series_length is None:
            raise Exception("Fit must be called before inverting")

        return Crop(self.length, self.time_series_length + self.length)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={self.length})"


class PadBack(Transform):
    """Class to pad the time series data with zeros at the end.

    This class has 2 different modes. This first mode is for padding at the end a specific number of zeros.
    The second pads the time series to a specific length by calculating the difference between the length of
    the time series and the desired length.

    Args:
        length: The length of the padding to be added.
        fixed_length: If True, the length is the time series will have the given length, after transforming.
        value: The value the time series is padded with. Defaults to 0.

    Examples:
    This example demonstrates how to pad 2 zeros at the end of the time series.

    >>> import torch
    >>> time_series = time_series = torch.arange(10).repeat(10, 1, 1).float()
    >>> pad = PadBack(2)
    >>> pad.fit(time_series)
    >>> padded_data = pad(time_series)
    >>> padded_data.shape
    torch.Size([10, 1, 12])
    >>> padded_data[0]
    tensor([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 0.]])

    Here the time series should be padded to a lenght of 15, the transform calculates, how many zeros to pad.

    >>> import torch
    >>> time_series = time_series = torch.arange(10).repeat(10, 1, 1).float()
    >>> pad = PadBack(15, True)
    >>> pad.fit(time_series)
    >>> padded_data = pad(time_series)
    >>> padded_data.shape
    torch.Size([10, 1, 15])
    >>> padded_data[0]
    tensor([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 0., 0., 0., 0.]])
    """

    def __init__(self, length: int, fixed_length: bool = False, value: float = 0) -> None:
        super().__init__()
        self.time_series_length: int | None = None
        self.fixed_length = fixed_length
        self.length: int = length
        self.value = value

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Fit the transformation by determining the length of the time series data.

        Args:
            time_series : The input time series data.
            targets: The target data.
        """
        self.time_series_length = time_series.shape[-1]
        if self.fixed_length:
            if self.length < self.time_series_length:
                raise RuntimeError("Pad length is less than the time series length")
            self.length = self.length - self.time_series_length

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the transformation by padding the time series data with zeros.

        Args:
            time_series: The input time series data.
            targets: The target data.

        Returns:
            The transformed time series data and targets.

        Raises:
            Exception: If the fit method has not been called before transforming.
        """
        if self.time_series_length is None:
            raise Exception("Fit must be called before transforming")
        if self.length == 0:
            return time_series, targets

        padding = torch.zeros((time_series.shape[0], time_series.shape[1], self.length)) + self.value
        return torch.cat([time_series, padding], dim=2), targets

    def _invert(self) -> Transform:
        """Invert the transformation by returning a Crop transform.

        Returns:
            The inverted transformation.

        Raises:
            Exception: If the fit method has not been called before inverting.
        """
        if self.time_series_length is None:
            raise Exception("Fit must be called before inverting")

        return Crop(0, self.time_series_length)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={self.length})"


class Filter(Transform):
    """Class to filter time series data based on a given filter function.

    This is a filter transform. The filter function is called on each time series of the dataset. When the
    filter method returns True, the time series says in the dataset. If False is returned, the time series
    is removed.

    Args:
        filter: The filter function.

    Note:
        The given function for filtering, has to take 2 parameters. Even if there are no targets involved in
        the training, None is passed through the transforms for the targets. The function is then called with
        function(data, None).

    Examples:
    In this example each second row gets multiplied by 2. Next, using the Filter transform, each time series
    with a value bigger than 13 gets filtered.

    >>> import torch
    >>> time_series = torch.arange(10).repeat(10, 1, 1).float()
    >>> time_series[0::2] = time_series[0::2] * 2
    >>> filter_lambda_expression = lambda x, y: True if torch.max(x) < 13 else False
    >>> filter = Filter(filter_lambda_expression)
    >>> filtered_data = filter(time_series)
    >>> filtered_data.shape
    torch.Size([5, 1, 10])
    >>> torch.max(filtered_data) > 13
    tensor(False)
    """

    def __init__(self, filter: Callable) -> None:
        super().__init__(True)
        self.filter: Callable = filter

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Fit the filter transformation to the given time series data.

        Args:
            time_series: The input time series data.
            targets: The target values associated with the time series data.

        """

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the filter transformation to the given time series data.

        Args:
            time_series: The input time series data.
            targets: The target values associated with the time series data.


        Returns:
            The filtered time series data and the filtered target values (if provided).

        """
        indecies = []
        if targets is None:
            for i in range(len(time_series)):
                if self.filter(time_series[i], None):
                    indecies.append(i)
            return time_series[indecies], None
        else:
            for i in range(len(time_series)):
                if self.filter(time_series[i], targets[i]):
                    indecies.append(i)
            return time_series[indecies], targets[indecies]

    def _invert(self) -> Transform:
        """Invert the filter transformation.

        Returns:
            The inverted transformation.

        Raises:
            NoInverseError: If the filter transformation is not invertible.

        """
        error_message = (
            "The filter transform can not be inverted. To invert this transform the whole datset"
            "has to be saved and a difference needs to be determined."
        )
        raise NoInverseError(error_message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SlidingWindow(Transform):
    """Class for applying sliding window segmentation to time series data.

    The SlidingWindow transform, cuts the time series into smaller pieces. The given window_size determins
    the size of the parts. The step_size describes, how many time steps are between the different windows.

    Args:
        window_size: The size of the sliding window.
        step_size: The step size between consecutive windows.

    Examples:
    This example cuts the time series in pairs of numbers.

    >>> import torch
    >>> time_series = torch.arange(10).repeat(10, 1, 1).float()
    >>> sliding_window = SlidingWindow(2)
    >>> transformed_data = sliding_window(time_series)
    >>> transformed_data.shape
    torch.Size([90, 1, 2])
    >>> transformed_data[0]
    tensor([[0., 1.]])

    This example cuts the time series in pairs of numbers with a step size of 4.

    >>> import torch
    >>> time_series = torch.arange(10).repeat(10, 1, 1).float()
    >>> sliding_window = SlidingWindow(3, 4)
    >>> transformed_data = sliding_window(time_series)
    >>> transformed_data.shape
    torch.Size([20, 1, 3])
    >>> transformed_data[0]
    tensor([[0., 1., 2.]])
    >>> transformed_data[1]
    tensor([[4., 5., 6.]])

    """

    def __init__(self, window_size: int, step_size: int = 1) -> None:
        super().__init__(True)
        self.window_size = window_size
        self.step_size = step_size

    def _fit(self, time_series: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Fit the sliding window transform to the given time series data.

        This method does not perform any fitting as the identity transformation does not
        require any parameters.

        Args:
            time_series: The input time series data.
            targets: The target values associated with the time series data.

        """

    def _transform(
        self, time_series: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the sliding window transform to the given time series data.

        Args:
            time_series: The input time series data.
            targets: The target values associated with the time series data.

        Returns:
            A tuple containing the transformed time series data
            and the transformed target values (if provided).

        """
        num_time_series, dimensions, time_steps = time_series.shape
        num_segments = (time_steps - self.window_size) // self.step_size + 1

        ts_segments = []
        targets_segmented = []
        for i in range(num_time_series):
            current_time_series = time_series[i]

            for j in range(num_segments):
                start_index = j * self.step_size
                end_index = start_index + self.window_size
                ts_segment = current_time_series[:, start_index:end_index]
                ts_segments.append(ts_segment.unsqueeze(0))

                if targets is not None:
                    targets_segmented.append(targets[i])

        ts_tensor = torch.cat(ts_segments, dim=0)

        if targets is None:
            targets_tensor = None
        else:
            targets_tensor = torch.tensor(targets_segmented)

        return ts_tensor, targets_tensor

    def _invert(self) -> Transform:
        """Invert the sliding window transform.

        Raises:
            NoInverseError: The sliding window transform does not have an inverse.
        """
        error_message = (
            "The SlidingWindow Transform cuts the time series into smaller peaced. This can not" "be undone."
        )
        raise NoInverseError(error_message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(window_size={self.window_size}, step_size={self.step_size})"
