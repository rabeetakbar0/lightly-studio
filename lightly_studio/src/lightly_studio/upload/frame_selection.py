"""Chooses which timestamps of a video to extract as still frames.

The selection samples the start and the end of a video densely, at a fixed interval, and
the middle sparsely, with a fixed number of evenly spaced frames. This fits videos where the
interesting events cluster at the start and the end, such as a gate opening and closing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSelectionConfig:
    """Parameters that control which timestamps of a video get extracted.

    Attributes:
        edge_window_seconds: Length of the dense window at the start and at the end.
        edge_interval_seconds: Spacing between frames inside the dense windows.
        middle_frames: Number of evenly spaced frames taken from the middle section.
        max_frames: Upper bound on the total number of frames. The selection is thinned
            evenly when it exceeds this bound.
    """

    edge_window_seconds: float = 20.0
    edge_interval_seconds: float = 1.0
    middle_frames: int = 35
    max_frames: int = 100

    def validate(self) -> None:
        """Raises ValueError when a parameter is outside its valid range."""
        if self.edge_window_seconds < 0:
            raise ValueError("edge_window_seconds must be >= 0")
        if self.edge_interval_seconds <= 0:
            raise ValueError("edge_interval_seconds must be > 0")
        if self.middle_frames < 0:
            raise ValueError("middle_frames must be >= 0")
        if self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")


def select_timestamps(duration_seconds: float, config: FrameSelectionConfig) -> list[float]:
    """Returns the sorted, unique timestamps in seconds to extract from a video.

    Args:
        duration_seconds: Length of the video. A non-positive duration yields a single
            frame at zero.
        config: The selection parameters.

    Returns:
        Timestamps in ascending order, each within [0, duration_seconds], at most
        config.max_frames long.
    """
    config.validate()
    if duration_seconds <= 0:
        return [0.0]

    edge = min(config.edge_window_seconds, duration_seconds / 2)
    start_window = _interval_timestamps(start=0.0, end=edge, interval=config.edge_interval_seconds)
    end_window = _interval_timestamps(
        start=duration_seconds - edge, end=duration_seconds, interval=config.edge_interval_seconds
    )
    middle = _evenly_spaced(start=edge, end=duration_seconds - edge, count=config.middle_frames)

    timestamps = sorted({round(t, 3) for t in start_window + middle + end_window})
    return _thin_evenly(values=timestamps, max_count=config.max_frames)


def _interval_timestamps(start: float, end: float, interval: float) -> list[float]:
    """Timestamps from start up to and including end, spaced by interval."""
    if end < start:
        return []
    count = int((end - start) / interval) + 1
    timestamps = [start + i * interval for i in range(count)]
    if timestamps[-1] < end - 1e-9:
        timestamps.append(end)
    return timestamps


def _evenly_spaced(start: float, end: float, count: int) -> list[float]:
    """Count timestamps strictly between start and end, evenly spaced."""
    if count <= 0 or end <= start:
        return []
    step = (end - start) / (count + 1)
    return [start + step * (i + 1) for i in range(count)]


def _thin_evenly(values: list[float], max_count: int) -> list[float]:
    """Keeps at most max_count values, spread evenly over the input, always keeping the ends."""
    if len(values) <= max_count:
        return values
    if max_count == 1:
        return [values[0]]
    last = len(values) - 1
    indices = {round(i * last / (max_count - 1)) for i in range(max_count)}
    return [values[i] for i in sorted(indices)]
