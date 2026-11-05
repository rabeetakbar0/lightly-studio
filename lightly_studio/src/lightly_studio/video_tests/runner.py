"""Starts, watches and stops TestApp pipeline runs on uploaded videos.

A run saves the uploaded videos to <uploads>/<name>/ and runs the TestApp's
App/test_pipeline.py with the TestApp interpreter (LIGHTLY_STUDIO_TESTAPP_PYTHON) from the
TestApp folder (LIGHTLY_STUDIO_TESTAPP_DIR). The scripts read the model and the folders from
GATEARM_MODEL_PATH, GATEARM_INPUT_DIR and GATEARM_OUTPUT_DIR. Processed videos, the log and
run.json go to <uploads>/results/<name>/. One run at a time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from lightly_studio.dataset import env
from lightly_studio.pipeline_training import model_catalog
from lightly_studio.pipeline_training import runs as training_runs
from lightly_studio.pipeline_training.model_catalog import ModelInfo
from lightly_studio.video_tests import log_progress, runs
from lightly_studio.video_tests.log_progress import VideoTestProgress
from lightly_studio.video_tests.runs import VideoEntry, VideoTestRun

PIPELINE_SCRIPT = Path("App") / "test_pipeline.py"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")
STOP_TIMEOUT_SECONDS = 10
_NAME_FORMAT = "%Y%m%d-%H%M%S"


@dataclass
class VideoTestEnvironment:
    """Whether a video test can start on this machine, and where its files go.

    Attributes:
        testapp_dir: The TestApp checkout, or None when not configured.
        python: The TestApp interpreter, or None when not configured.
        uploads_dir: Where uploaded videos go, one folder per run.
        results_dir: Where processed videos go, one folder per run.
        problems: What is missing; a test can start when this is empty.
    """

    testapp_dir: str | None
    python: str | None
    uploads_dir: str | None
    results_dir: str | None
    problems: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True when nothing is missing."""
        return not self.problems


class VideoTestRunner:
    """The video tests of this backend process."""

    def __init__(self) -> None:
        """Starts with no runs; existing result folders are picked up on first use."""
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._stop_requested: set[str] = set()
        self._starting = False
        self._lock = threading.Lock()
        self._reconciled = False

    def environment(self) -> VideoTestEnvironment:
        """Checks the configured TestApp folder and interpreter."""
        testapp_dir = env.LIGHTLY_STUDIO_TESTAPP_DIR
        python = env.LIGHTLY_STUDIO_TESTAPP_PYTHON
        problems = []
        if testapp_dir is None:
            problems.append("LIGHTLY_STUDIO_TESTAPP_DIR is not set.")
        elif not (testapp_dir / PIPELINE_SCRIPT).is_file():
            problems.append(f"{testapp_dir} has no {PIPELINE_SCRIPT}.")
        if python is None:
            problems.append("LIGHTLY_STUDIO_TESTAPP_PYTHON is not set.")
        elif shutil.which(python) is None and not Path(python).is_file():
            problems.append(f"TestApp interpreter {python} does not exist.")
        uploads_dir = _uploads_dir()
        return VideoTestEnvironment(
            testapp_dir=str(testapp_dir) if testapp_dir else None,
            python=python,
            uploads_dir=str(uploads_dir) if uploads_dir else None,
            results_dir=str(uploads_dir / runs.RESULTS_FOLDER) if uploads_dir else None,
            problems=problems,
        )

    def list_runs(self) -> list[VideoTestRun]:
        """All video tests, newest first, with live progress for a running one."""
        self._reconcile()
        uploads_dir = _uploads_dir()
        if uploads_dir is None:
            return []
        records = runs.list_records(results_dir=uploads_dir / runs.RESULTS_FOLDER)
        for record in records:
            if record.is_active:
                runs.apply_progress(record=record, progress=self.progress(name=record.name))
        return records

    def get_run(self, name: str) -> VideoTestRun:
        """One video test, with live progress while it runs.

        Raises:
            ValueError: If there is no such run.
        """
        self._reconcile()
        record = runs.read_record(run_dir=self._results_dir(name=name))
        if record is None:
            raise ValueError(f"Video test '{name}' does not exist.")
        if record.is_active:
            runs.apply_progress(record=record, progress=self.progress(name=name))
        return record

    def progress(self, name: str) -> VideoTestProgress:
        """Where the run is, read from its log."""
        return runs.read_progress(run_dir=self._results_dir(name=name))

    def read_log(self, name: str, lines: int) -> str:
        """The last lines of the run's log.

        Raises:
            ValueError: If there is no such run.
        """
        self.get_run(name=name)
        return training_runs.read_log_tail(run_dir=self._results_dir(name=name), lines=lines)

    def start(self, model: str, files: Sequence[tuple[str, BinaryIO]]) -> VideoTestRun:
        """Saves the videos to a new upload folder and starts the pipeline on them.

        Args:
            model: A model of the models folder.
            files: (file name, readable stream) per uploaded video.

        Raises:
            ValueError: If the environment is not ready, the model does not exist, a file is
                not a video, there are no files, or a video test is already running.
        """
        environment = self.environment()
        if not environment.ready:
            raise ValueError(" ".join(environment.problems))
        assert environment.testapp_dir is not None
        assert environment.python is not None
        info = _model(name=model)
        names = _file_names(names=[name for name, _ in files])
        with self._lock:
            if self._starting or any(p.poll() is None for p in self._processes.values()):
                raise ValueError("A video test is already running; wait for it or stop it.")
            self._starting = True
        try:
            uploads_dir = _uploads_dir()
            assert uploads_dir is not None
            name = _new_name(uploads_dir=uploads_dir)
            input_dir = uploads_dir / name
            output_dir = uploads_dir / runs.RESULTS_FOLDER / name
            videos = _save_files(
                input_dir=input_dir, files=[(n, f) for n, (_, f) in zip(names, files)]
            )
            output_dir.mkdir(parents=True)
            record = VideoTestRun(
                name=name,
                model=info.name,
                model_path=str(info.onnx_path.resolve()),
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                status=runs.STATUS_RUNNING,
                created_at=training_runs.now(),
                command=[environment.python, "-u", str(PIPELINE_SCRIPT)],
                videos=videos,
            )
            return self._launch(record=record, testapp_dir=Path(environment.testapp_dir))
        finally:
            with self._lock:
                self._starting = False

    def stop(self, name: str) -> VideoTestRun:
        """Terminates the pipeline; the run ends as 'stopped'.

        Raises:
            ValueError: If there is no such run or it is not running.
        """
        record = self.get_run(name=name)
        with self._lock:
            process = self._processes.get(name)
            if process is None or process.poll() is not None:
                raise ValueError(f"Video test '{name}' is not running.")
            self._stop_requested.add(name)
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
        record.status = runs.STATUS_STOPPED
        return record

    def _launch(self, record: VideoTestRun, testapp_dir: Path) -> VideoTestRun:
        output_dir = Path(record.output_dir)
        child_env = {
            **os.environ,
            "GATEARM_MODEL_PATH": record.model_path,
            "GATEARM_INPUT_DIR": record.input_dir,
            "GATEARM_OUTPUT_DIR": record.output_dir,
            "PYTHONUNBUFFERED": "1",
        }
        log = (output_dir / runs.LOG_FILE).open("ab")
        try:
            process = subprocess.Popen(
                record.command,
                cwd=testapp_dir,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        finally:
            log.close()
        record.pid = process.pid
        with self._lock:
            self._processes[record.name] = process
        runs.write_record(run_dir=output_dir, record=record)
        threading.Thread(target=self._supervise, args=(record, process), daemon=True).start()
        return record

    def _supervise(self, record: VideoTestRun, process: subprocess.Popen[bytes]) -> None:
        output_dir = Path(record.output_dir)
        record.exit_code = process.wait()
        with self._lock:
            self._processes.pop(record.name, None)
            stopped = record.name in self._stop_requested
            self._stop_requested.discard(record.name)
        runs.apply_progress(record=record, progress=runs.read_progress(run_dir=output_dir))
        if stopped:
            record.status = runs.STATUS_STOPPED
        elif record.exit_code != 0:
            record.status = runs.STATUS_FAILED
            record.error = f"The pipeline exited with code {record.exit_code}; see the log."
        elif not any(v.status == log_progress.VIDEO_DONE for v in record.videos):
            record.status = runs.STATUS_FAILED
            record.error = "No video was processed; see the log."
        else:
            record.status = runs.STATUS_FINISHED
        for video in record.videos:
            if video.status == log_progress.VIDEO_PROCESSING:
                video.status = log_progress.VIDEO_FAILED
        record.finished_at = training_runs.now()
        runs.write_record(run_dir=output_dir, record=record)

    def _reconcile(self) -> None:
        """Marks runs left running by an earlier backend process as failed, once."""
        with self._lock:
            if self._reconciled:
                return
            self._reconciled = True
        uploads_dir = _uploads_dir()
        if uploads_dir is None:
            return
        for record in runs.list_records(results_dir=uploads_dir / runs.RESULTS_FOLDER):
            if record.is_active and record.name not in self._processes:
                record.status = runs.STATUS_FAILED
                record.error = "The backend was restarted while the video test was running."
                record.finished_at = training_runs.now()
                runs.write_record(run_dir=Path(record.output_dir), record=record)

    def _results_dir(self, name: str) -> Path:
        uploads_dir = _uploads_dir()
        if uploads_dir is None:
            raise ValueError("LIGHTLY_STUDIO_TESTAPP_DIR is not set.")
        return uploads_dir / runs.RESULTS_FOLDER / training_runs.validate_name(name=name)


def _uploads_dir() -> Path | None:
    if env.LIGHTLY_STUDIO_TESTAPP_UPLOADS_DIR is not None:
        return env.LIGHTLY_STUDIO_TESTAPP_UPLOADS_DIR
    if env.LIGHTLY_STUDIO_TESTAPP_DIR is not None:
        return env.LIGHTLY_STUDIO_TESTAPP_DIR / "uploads"
    return None


def _model(name: str) -> ModelInfo:
    for info in model_catalog.list_models(models_dir=env.LIGHTLY_STUDIO_MODELS_DIR):
        if info.name == name:
            return info
    raise ValueError(f"Model '{name}' is not in the models folder.")


def _file_names(names: list[str]) -> list[str]:
    """Safe, unique file names for the uploads, in the given order.

    Raises:
        ValueError: If there are no files or a file is not a video.
    """
    if not names:
        raise ValueError("Add at least one video.")
    not_videos = [n for n in names if Path(n).suffix.lower() not in VIDEO_EXTENSIONS]
    if not_videos:
        raise ValueError(
            f"Not a video ({', '.join(VIDEO_EXTENSIONS)}): {', '.join(sorted(not_videos))}."
        )
    result: list[str] = []
    for name in names:
        path = Path(Path(name).name)
        candidate, number = path.name, 2
        while candidate in result:
            candidate = f"{path.stem} ({number}){path.suffix}"
            number += 1
        result.append(candidate)
    return result


def _new_name(uploads_dir: Path) -> str:
    """The local start time as the run name, with -2, -3... when that folder exists."""
    base = datetime.now(timezone.utc).astimezone().strftime(_NAME_FORMAT)
    name, number = base, 2
    while (uploads_dir / name).exists() or (uploads_dir / runs.RESULTS_FOLDER / name).exists():
        name = f"{base}-{number}"
        number += 1
    return name


def _save_files(input_dir: Path, files: list[tuple[str, BinaryIO]]) -> list[VideoEntry]:
    """Writes the uploads into a new folder; removes the folder again when a write fails."""
    input_dir.mkdir(parents=True)
    try:
        videos = []
        for name, stream in files:
            target = input_dir / name
            with target.open("wb") as out:
                shutil.copyfileobj(stream, out, length=8 * 1024 * 1024)
            videos.append(VideoEntry(file_name=name, size_bytes=target.stat().st_size))
        return videos
    except Exception:
        shutil.rmtree(input_dir, ignore_errors=True)
        raise


video_test_runner = VideoTestRunner()
