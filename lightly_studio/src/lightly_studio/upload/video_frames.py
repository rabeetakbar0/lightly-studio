"""Extracts still frames from a video file at chosen timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from lightly_studio.upload import frame_selection
from lightly_studio.upload.frame_selection import FrameSelectionConfig


@dataclass(frozen=True)
class ExtractedFrame:
    """One frame written to disk.

    Attributes:
        path: Location of the JPEG file.
        frame_number: Index of the frame inside the video, starting at zero.
        timestamp_seconds: Position of the frame inside the video.
    """

    path: Path
    frame_number: int
    timestamp_seconds: float


def get_duration_seconds(video_path: Path) -> float:
    """Returns the duration of a video, read from its container metadata.

    Raises:
        ValueError: If the file cannot be opened as a video.
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path.name}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"Cannot read the duration of video file: {video_path.name}")
        return frame_count / fps
    finally:
        capture.release()


def extract_frames(
    video_path: Path, output_dir: Path, config: FrameSelectionConfig
) -> list[ExtractedFrame]:
    """Writes the frames selected by config to output_dir as JPEG files.

    Files are named `<video stem>_f<frame number>.jpg`. A timestamp that the decoder
    cannot seek to, for example one past the last decodable frame, is skipped.

    Args:
        video_path: The video to read.
        output_dir: Directory for the JPEG files. It is created when missing.
        config: Which timestamps to extract.

    Returns:
        The written frames in ascending frame order, without duplicates.

    Raises:
        ValueError: If the file cannot be opened as a video.
    """
    duration = get_duration_seconds(video_path=video_path)
    timestamps = frame_selection.select_timestamps(duration_seconds=duration, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path.name}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames: list[ExtractedFrame] = []
        seen_frame_numbers: set[int] = set()
        for timestamp in timestamps:
            frame_number = round(timestamp * fps)
            if frame_number in seen_frame_numbers:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, image = capture.read()
            if not ok:
                continue
            seen_frame_numbers.add(frame_number)
            path = output_dir / f"{video_path.stem}_f{frame_number:06d}.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames.append(
                ExtractedFrame(
                    path=path, frame_number=frame_number, timestamp_seconds=frame_number / fps
                )
            )
        return frames
    finally:
        capture.release()
