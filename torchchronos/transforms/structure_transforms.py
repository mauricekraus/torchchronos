from typing import Optional, Callable

import torch

from .base_transforms import Transform
from .transformation_exceptions import NoInverseError

"""
Structure transforms manipulate the structure of the underlying data.
Examples are cropping, and Padding.
"""


class Crop(Transform):
    """
    Crop transformation that crops a given portion of the time series.

    """

    def __init__(self, start: int, end: int) -> None:
        """
        Initializes a new instance of the Crop class.

        Args:
            start (int): The starting index of the crop.
            end (int): The ending index of the crop.
        """
        super().__init__()
        self.start = start
        self.end = end

    def _fit(self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        """
        Fit the crop transformation.

        Args:
            time_series (torch.Tensor): The input time series.
            targets (torch.Tensor, optional): The target values associated with the time series.

        Raises:
            ValueError: If start or end is a negative integer, or if start is greater than or equal to end,
                        or if end is greater than the length of the time series.

        Returns:
            None

        """
        if self.start < 0 or self.end < 0:
            raise ValueError("Start and end must be positive integers")
        if self.start >= self.end:
            raise ValueError("Start must be less than end")
        if self.end > time_series.shape[-1]:
            raise ValueError("End must be less than the length of the time series")

    def _transform(
        self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply the crop transformation to the input time series.

        Args:
            time_series (torch.Tensor): The input time series.
            targets (torch.Tensor, optional): The target values associated with the time series.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: The cropped time series and the targets (unchanged).

        """
        return time_series[:, :, self.start : self.end], targets

    def _invert(self) -> Transform:
        """
        Invert the crop transformation.

        Raises:
            NoInverseError: If the crop transformation is not invertible.

        """
        raise NoInverseError("Crop transformation is not invertible")

    def __repr__(self) -> str:
        """
        Return a string representation of the Crop object.

        Returns:
            str: The string representation of the Crop object.

        """
        return f"Crop(start={self.start}, end={self.end})"


class PadFront(Transform):
    """
    A transformation that pads the front of the time series with zeros.

    Attributes:
        length (int): The length of the padding to be added.
        time_series_length (int | None): The length of the time series.

    """

    def __init__(self, length: int) -> None:
        """
        Initializes a new instance of the PadFront class.

        Args:
            length (int): The length of the padding to be added.

        """
        super().__init__()
        self.length = length
        self.time_series_length: int | None = None

    def _fit(self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        """
        Fits the transformation by determining the length of the time series.

        Args:
            time_series (torch.Tensor): The input time series.
            targets (torch.Tensor, optional): The target values associated with the time series. Defaults to None.

        Returns:
            None
        """
        self.time_series_length = time_series.shape[-1]

    def _transform(
        self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Applies the transformation by padding the front of the time series with zeros.

        Args:
            time_series (torch.Tensor): The input time series.
            targets (torch.Tensor, optional): The target values associated with the time series. Defaults to None.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: The transformed time series and targets (if provided).
        """
        if self.time_series_length is None:
            raise Exception("Fit must be called before transforming")

        zeros = torch.zeros((time_series.shape[0], time_series.shape[1], self.length))
        return torch.cat([zeros, time_series], dim=2), targets

    def _invert(self) -> Transform:
        """
        Inverts the transformation by cropping the padded front.

        Returns:
            Crop: The inverted transformation.

        Raises:
            Exception: If the fit method has not been called before inverting.
        """
        if self.time_series_length is None:
            raise Exception("Fit must be called before inverting")

        return Crop(self.length, self.time_series_length + self.length)

    def __repr__(self) -> str:
        """
        Returns a string representation of the PadFront object.

        Returns:
            str: The string representation of the PadFront object.
        """
        return f"{self.__class__.__name__}(length={self.length})"


class PadBack(Transform):
    """
    A transformation that pads the time series data with zeros at the end.

    Attributes:
        length (int): The length of the padding to be added.
        time_series_length (int | None): The length of the time series data.
    """

    def __init__(self, length: int) -> None:
        """
        Initializes a new instance of the PadBack class.

        Args:
            length (int): The length of the padding to be added.
        """
        super().__init__()
        self.length = length
        self.time_series_length: int | None = None

    def _fit(self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        """
        Fits the transformation by determining the length of the time series data.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target data. Defaults to None.
        """
        self.time_series_length = time_series.shape[-1]

    def _transform(
        self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Applies the transformation by padding the time series data with zeros.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (Optional[torch.Tensor], optional): The target data. Defaults to None.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: The transformed time series data and targets.

        Raises:
            Exception: If the fit method has not been called before transforming.
        """
        if self.time_series_length is None:
            raise Exception("Fit must be called before transforming")

        zeros = torch.zeros((time_series.shape[0], time_series.shape[1], self.length))
        return torch.cat([time_series, zeros], dim=2), targets

    def _invert(self) -> Transform:
        """
        Inverts the transformation by returning a Crop transform.

        Returns:
            Transform: The inverted transformation.

        Raises:
            Exception: If the fit method has not been called before inverting.
        """
        if self.time_series_length is None:
            raise Exception("Fit must be called before inverting")

        return Crop(0, self.time_series_length)

    def __repr__(self) -> str:
        """
        Returns a string representation of the PadBack object.

        Returns:
            str: The string representation of the object.
        """
        return f"{self.__class__.__name__}(length={self.length})"


class Filter(Transform):
    """
    A transformation that filters time series data based on a given filter function.

    Attributes:
        filter (Callable): The filter function.

    """

    def __init__(self, filter: Callable) -> None:
        """
        Initializes a new instance of the Filter class.

        Args:
            filter (Callable): The filter function to be applied to the time series data.

        """
        super().__init__(True)
        self.filter: Callable = filter

    def _fit(self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        """
        Fits the filter transformation to the given time series data.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target values associated with the time series data. Defaults to None.

        Returns:
            None
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Applies the filter transformation to the given time series data.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target values associated with the time series data. Defaults to None.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: The filtered time series data and the filtered target values (if provided).

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
        """
        Inverts the filter transformation.

        Returns:
            Transform: The inverted transformation.

        Raises:
            NoInverseError: If the filter transformation is not invertible.

        """
        raise NoInverseError("Filter transformation is not invertible")

    def __repr__(self) -> str:
        """
        Returns a string representation of the Filter object.

        Returns:
            str: The string representation of the object.

        """
        return f"{self.__class__.__name__}()"


class SlidingWindow(Transform):
    """
    A transform that applies sliding window segmentation to time series data.

    Attributes:
        window_size (int): The size of the sliding window.
        step_size (int): The step size between consecutive windows.
    """

    def __init__(self, window_size: int, step_size: int) -> None:
        """
        Initializes a new instance of the SlidingWindow class.

        Args:
            window_size (int): The size of the sliding window.
            step_size (int): The step size between consecutive windows.
        """
        super().__init__(True)
        self.window_size = window_size
        self.step_size = step_size

    def _fit(self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        """
        Fit the sliding window transform to the given time series data.

        This method does not perform any fitting as the identity transformation does not require any parameters.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target values associated with the time series data.

        Returns:
            None
        """
        pass

    def _transform(
        self, time_series: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply the sliding window transform to the given time series data.

        Args:
            time_series (torch.Tensor): The input time series data.
            targets (torch.Tensor, optional): The target values associated with the time series data.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the transformed time series data
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
        """
        Invert the sliding window transform.

        Raises:
            NoInverseError: The sliding window transform does not have an inverse.
        """
        raise NoInverseError()

    def __repr__(self) -> str:
        """
        Return a string representation of the SlidingWindow transform.

        Returns:
            str: A string representation of the SlidingWindow transform.
        """
        return f"{self.__class__.__name__}(window_size={self.window_size}, step_size={self.step_size})"
