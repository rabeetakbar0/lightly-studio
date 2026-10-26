from typing import Any

import pytest

from lightly_studio.upload import frame_selection
from lightly_studio.upload.frame_selection import FrameSelectionConfig


class TestFrameSelectionConfig:
    def test_validate(self) -> None:
        FrameSelectionConfig().validate()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"edge_window_seconds": -1.0},
            {"edge_interval_seconds": 0.0},
            {"middle_frames": -1},
            {"max_frames": 0},
        ],
    )
    def test_validate__invalid(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match=r"must be"):
            FrameSelectionConfig(**kwargs).validate()


def test_select_timestamps() -> None:
    # 100 s video: 0..20 s every 1 s (21 frames), 80..100 s every 1 s (21 frames),
    # and 3 middle frames evenly spaced strictly inside 20..80 s, at 35, 50 and 65 s.
    config = FrameSelectionConfig(
        edge_window_seconds=20.0, edge_interval_seconds=1.0, middle_frames=3, max_frames=1000
    )
    timestamps = frame_selection.select_timestamps(duration_seconds=100.0, config=config)
    assert timestamps[:3] == [0.0, 1.0, 2.0]
    assert timestamps[-3:] == [98.0, 99.0, 100.0]
    assert 35.0 in timestamps
    assert 50.0 in timestamps
    assert 65.0 in timestamps
    assert len(timestamps) == 21 + 21 + 3
    assert timestamps == sorted(timestamps)


def test_select_timestamps__short_video_is_all_edge() -> None:
    # 10 s video with 20 s windows: windows shrink to 5 s each, no middle section.
    config = FrameSelectionConfig(
        edge_window_seconds=20.0, edge_interval_seconds=2.5, middle_frames=35, max_frames=1000
    )
    timestamps = frame_selection.select_timestamps(duration_seconds=10.0, config=config)
    assert timestamps == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_select_timestamps__max_frames_thins_evenly() -> None:
    config = FrameSelectionConfig(
        edge_window_seconds=20.0, edge_interval_seconds=1.0, middle_frames=35, max_frames=5
    )
    timestamps = frame_selection.select_timestamps(duration_seconds=100.0, config=config)
    assert len(timestamps) == 5
    assert timestamps[0] == 0.0
    assert timestamps[-1] == 100.0


def test_select_timestamps__zero_duration() -> None:
    timestamps = frame_selection.select_timestamps(
        duration_seconds=0.0, config=FrameSelectionConfig()
    )
    assert timestamps == [0.0]


def test_select_timestamps__middle_count_independent_of_length() -> None:
    config = FrameSelectionConfig(
        edge_window_seconds=20.0, edge_interval_seconds=1.0, middle_frames=35, max_frames=1000
    )
    short = frame_selection.select_timestamps(duration_seconds=120.0, config=config)
    long = frame_selection.select_timestamps(duration_seconds=3600.0, config=config)
    # 21 start + 21 end + 35 middle, no overlaps for either length.
    assert len(short) == 77
    assert len(long) == 77
