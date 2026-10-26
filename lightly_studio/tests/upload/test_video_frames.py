from pathlib import Path

import cv2
import numpy as np
import pytest

from lightly_studio.upload import video_frames
from lightly_studio.upload.frame_selection import FrameSelectionConfig

FPS = 10
FRAME_COUNT = 100  # 10 s video.


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (64, 48))
    for i in range(FRAME_COUNT):
        frame = np.full((48, 64, 3), fill_value=i * 2, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_get_duration_seconds(video_path: Path) -> None:
    duration = video_frames.get_duration_seconds(video_path=video_path)
    assert duration == pytest.approx(FRAME_COUNT / FPS)


def test_get_duration_seconds__not_a_video(tmp_path: Path) -> None:
    path = tmp_path / "not_a_video.mp4"
    path.write_bytes(b"hello")
    with pytest.raises(ValueError, match=r"not_a_video\.mp4"):
        video_frames.get_duration_seconds(video_path=path)


def test_extract_frames(video_path: Path, tmp_path: Path) -> None:
    # 10 s video, 2 s edge windows at 1 s steps, 2 middle frames: 0,1,2 | 4,6 | 8,9,10 s.
    config = FrameSelectionConfig(
        edge_window_seconds=2.0, edge_interval_seconds=1.0, middle_frames=2, max_frames=100
    )
    output_dir = tmp_path / "frames"
    frames = video_frames.extract_frames(
        video_path=video_path, output_dir=output_dir, config=config
    )
    frame_numbers = [f.frame_number for f in frames]
    # Frame 100 is past the last frame (index 99) and is skipped.
    assert frame_numbers == [0, 10, 20, 40, 60, 80, 90]
    assert all(f.path.exists() for f in frames)
    assert frames[1].path.name == "clip_f000010.jpg"
    assert frames[1].timestamp_seconds == pytest.approx(1.0)
    # Decoded frames carry the brightness written at that index, so the seek landed right.
    image = cv2.imread(str(frames[2].path))
    assert image.shape == (48, 64, 3)
    assert abs(int(image.mean()) - 40) <= 4


def test_extract_frames__max_frames(video_path: Path, tmp_path: Path) -> None:
    config = FrameSelectionConfig(
        edge_window_seconds=2.0, edge_interval_seconds=0.5, middle_frames=10, max_frames=4
    )
    frames = video_frames.extract_frames(
        video_path=video_path, output_dir=tmp_path / "frames", config=config
    )
    assert len(frames) <= 4
    assert frames[0].frame_number == 0
