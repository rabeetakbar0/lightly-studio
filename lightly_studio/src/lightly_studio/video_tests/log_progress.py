"""Reads the progress of a TestApp batch run from its log.

The TestApp scripts (App/test_pipeline.py and App/video_pipeline.py) log one line per video
("[2/3] clip.mp4 (12.3 MB)"), a progress line every 300 frames ("Progress 600/1800 (33%) |
... fps=61.2"), the benchmark stats and detection summary as indented "key: value" lines, and
"Done. Annotated video: <path>" or "Failed to process <path>" at the end of each video.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

VIDEO_PENDING = "pending"
VIDEO_PROCESSING = "processing"
VIDEO_DONE = "done"
VIDEO_FAILED = "failed"

_VIDEO = re.compile(r"\[(\d+)/(\d+)\] (.+) \([\d.]+ MB\)$")
_PROGRESS = re.compile(r"Progress (\d+)/(\d+) \(\d+%\).*fps=([\d.]+)")
_DONE = re.compile(r"Done\. Annotated video: (.+)$")
_FAILED = re.compile(r"Failed to process (.+)$")
_SECOND_PASS = "re-running the pipeline with backup locations"
_SECTIONS = {"Benchmark stats:": "benchmark", "Detection summary:": "summary"}
_SECTION_LINE = re.compile(r"^  (\w+): (.*)$")
_BATCH_FINISHED = "Batch finished:"


@dataclass
class VideoProgress:
    """The state of one video of the batch.

    Attributes:
        file_name: The uploaded file name.
        status: pending, processing, done or failed.
        output_path: The processed video, once it is written.
        summary: The detection summary the pipeline printed, as text.
        benchmark: The benchmark stats the pipeline printed, as text.
    """

    file_name: str
    status: str = VIDEO_PENDING
    output_path: str | None = None
    summary: dict[str, str] = field(default_factory=dict)
    benchmark: dict[str, str] = field(default_factory=dict)


@dataclass
class VideoTestProgress:
    """Where a batch run is.

    Attributes:
        total: Number of videos in the batch; 0 before the first video starts.
        current_index: 1-based index of the video being processed; 0 before the first.
        current_video: File name of that video.
        frames_done: Frames of the current pass of the current video.
        frames_total: Frames of the current video.
        fps: Processing speed of the last progress line.
        pass_number: 2 when the pipeline re-runs the video with the backup LED locations.
        videos: Every video seen so far, in processing order.
        batch_finished: True once the batch summary was logged.
    """

    total: int = 0
    current_index: int = 0
    current_video: str | None = None
    frames_done: int = 0
    frames_total: int = 0
    fps: float | None = None
    pass_number: int = 1
    videos: list[VideoProgress] = field(default_factory=list)
    batch_finished: bool = False

    @property
    def percent(self) -> float:
        """Share of the batch done, counting the current video by its frames."""
        if self.batch_finished:
            return 100.0
        if not self.total:
            return 0.0
        finished = sum(v.status in (VIDEO_DONE, VIDEO_FAILED) for v in self.videos)
        current = self.videos[-1] if self.videos else None
        share = 0.0
        if current is not None and current.status == VIDEO_PROCESSING and self.frames_total:
            share = min(self.frames_done / self.frames_total, 1.0)
        return round(100 * (finished + share) / self.total, 1)


def parse_progress(text: str) -> VideoTestProgress:
    """Parses a whole log; unknown lines are ignored."""
    progress = VideoTestProgress()
    section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        current = progress.videos[-1] if progress.videos else None
        if section is not None and current is not None:
            item = _SECTION_LINE.match(line)
            if item:
                getattr(current, section)[item.group(1)] = item.group(2)
                continue
            section = None
        if (match := _VIDEO.search(line)) is not None:
            progress.current_index = int(match.group(1))
            progress.total = int(match.group(2))
            progress.current_video = match.group(3)
            progress.frames_done = 0
            progress.frames_total = 0
            progress.fps = None
            progress.pass_number = 1
            progress.videos.append(VideoProgress(file_name=match.group(3), status=VIDEO_PROCESSING))
        elif (match := _PROGRESS.search(line)) is not None:
            progress.frames_done = int(match.group(1))
            progress.frames_total = int(match.group(2))
            progress.fps = float(match.group(3))
        elif _SECOND_PASS in line:
            progress.pass_number = 2
            progress.frames_done = 0
        elif current is not None and (match := _DONE.search(line)) is not None:
            current.status = VIDEO_DONE
            current.output_path = match.group(1)
        elif current is not None and _FAILED.search(line) is not None:
            current.status = VIDEO_FAILED
        elif _BATCH_FINISHED in line:
            progress.batch_finished = True
        else:
            section = next((name for key, name in _SECTIONS.items() if line.endswith(key)), None)
    return progress
