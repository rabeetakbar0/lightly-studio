"""Video test records: one run.json per run in its results folder."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lightly_studio.video_tests import log_progress
from lightly_studio.video_tests.log_progress import VideoTestProgress

RUN_FILE = "run.json"
LOG_FILE = "log.txt"
RESULTS_FOLDER = "results"

STATUS_RUNNING = "running"
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"


@dataclass
class VideoEntry:
    """One uploaded video of a run.

    Attributes:
        file_name: The file name in the run's upload folder.
        size_bytes: Size of the upload.
        status: pending, processing, done or failed (see log_progress).
        output_path: The processed video, once written.
        summary: The detection summary the pipeline printed.
        benchmark: The benchmark stats the pipeline printed.
    """

    file_name: str
    size_bytes: int
    status: str = log_progress.VIDEO_PENDING
    output_path: str | None = None
    summary: dict[str, str] = field(default_factory=dict)
    benchmark: dict[str, str] = field(default_factory=dict)


@dataclass
class VideoTestRun:
    """The manifest of one video test.

    Attributes:
        name: The start time as YYYYmmdd-HHMMSS; the name of the upload and results folders.
        model: The model name in the models folder.
        model_path: The ONNX file the pipeline ran with.
        input_dir: The folder with the uploaded videos.
        output_dir: The folder with the processed videos, the log and this manifest.
        status: running, finished, failed or stopped.
        created_at: ISO timestamp of the start.
        finished_at: ISO timestamp of the end, None while running.
        pid: Process id of the pipeline while it runs.
        exit_code: Exit code of the pipeline once it ended.
        error: Why the run failed, when it did.
        command: The command as launched.
        videos: The uploaded videos with their results.
    """

    name: str
    model: str
    model_path: str
    input_dir: str
    output_dir: str
    status: str
    created_at: str
    finished_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    error: str | None = None
    command: list[str] = field(default_factory=list)
    videos: list[VideoEntry] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        """True while the pipeline runs."""
        return self.status == STATUS_RUNNING


def write_record(run_dir: Path, record: VideoTestRun) -> None:
    """Writes run.json atomically, so a reader never sees a partial file."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / f"{RUN_FILE}.tmp"
    tmp.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / RUN_FILE)


def read_record(run_dir: Path) -> VideoTestRun | None:
    """Reads run.json of a folder; None when the folder is not a video test."""
    path = run_dir / RUN_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["videos"] = [VideoEntry(**video) for video in data.get("videos", [])]
    return VideoTestRun(**data)


def list_records(results_dir: Path) -> list[VideoTestRun]:
    """All video tests, newest first."""
    if not results_dir.is_dir():
        return []
    records = [read_record(run_dir=folder) for folder in results_dir.iterdir() if folder.is_dir()]
    return sorted(
        (r for r in records if r is not None), key=lambda r: (r.created_at, r.name), reverse=True
    )


def read_progress(run_dir: Path) -> VideoTestProgress:
    """Parses the log of a run; empty progress when there is no log yet."""
    path = run_dir / LOG_FILE
    if not path.is_file():
        return VideoTestProgress()
    return log_progress.parse_progress(text=path.read_text(encoding="utf-8", errors="replace"))


def apply_progress(record: VideoTestRun, progress: VideoTestProgress) -> None:
    """Copies the per-video state of the log onto the record's videos, matched by name."""
    by_name = {video.file_name: video for video in progress.videos}
    for video in record.videos:
        seen = by_name.get(video.file_name)
        if seen is None:
            continue
        video.status = seen.status
        video.output_path = seen.output_path
        video.summary = dict(seen.summary)
        video.benchmark = dict(seen.benchmark)
