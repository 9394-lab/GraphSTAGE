from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class TimestampsOfDay(TimeFeature):
    """Dynamic timestamps of day, with the number of timestamps depending on the sampling frequency.
       Encoded as value between [-0.5, 0.5].
    """

    def __init__(self, num_timestamps):
        super().__init__()
        self.num_timestamps = num_timestamps

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # Compute the total number of seconds in a day (24 * 60 * 60)
        total_seconds_in_day = 24 * 60 * 60
        # Compute the timestamp position within the day as a fraction of total seconds
        relative_position = (index.hour * 3600 + index.minute * 60 + index.second) / total_seconds_in_day
        # Scale it to the range of [-0.5, 0.5]
        return (relative_position * self.num_timestamps / (self.num_timestamps - 1)) - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """
    offset = to_offset(freq_str)

    # Determine the number of timestamps for 'TimestampsOfDay'
    if isinstance(offset, offsets.Minute):
        num_timestamps = 24 * 60 // offset.n  # e.g., 288 for '5min'
    elif isinstance(offset, offsets.Hour):
        num_timestamps = 24 // offset.n  # e.g., 24 for '1h'
    else:
        num_timestamps = None  # Not used for other frequency types

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        # offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [lambda: TimestampsOfDay(num_timestamps), DayOfWeek, DayOfMonth, DayOfYear],

        offsets.Minute: [
            lambda: TimestampsOfDay(num_timestamps),  # Use TimestampsOfDay instead of MinuteOfHour and HourOfDay
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            # Check if the class is TimestampsOfDay and handle instantiation accordingly
            return [cls() if not callable(cls) else cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
