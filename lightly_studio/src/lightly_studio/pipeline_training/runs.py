"""Run folders: one folder per training run with a manifest, the exp file and the log."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightly_studio.pipeline_training import log_metrics
from lightly_studio.pipeline_training.log_metrics import TrainingProgress

RUN_FILE = "run.json"
EXP_FILE = "exp.py"
LOG_FILE = "log.txt"
EXPORT_DIR = "export"

STATUS_RUNNING = "running"
STATUS_EXPORTING = "exporting"
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"
ACTIVE_STATUSES = (STATUS_RUNNING, STATUS_EXPORTING)

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass
class RunRecord:
    """The manifest of one run, stored as run.json in its folder.

    Attributes:
        name: Folder name, also the name of the exported model.
        dataset: The prepared dataset the run trains on.
        classes: Class names in category id order; the exported model's class list.
        config: The TrainConfig as a dict.
        status: running, exporting, finished, failed or stopped.
        created_at: ISO timestamp of the start.
        finished_at: ISO timestamp of the end, None while active.
        pid: Process id of the training process while it runs.
        exit_code: Exit code of the training process once it ended.
        onnx_path: The exported model in the models folder, when export succeeded.
        error: Why the run failed, when it did.
        command: The train command as launched.
        dataset_info: The dataset summary at the start: counts, splits, annotation source.
    """

    name: str
    dataset: str
    classes: list[str]
    config: dict[str, Any]
    status: str
    created_at: str
    finished_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    onnx_path: str | None = None
    error: str | None = None
    command: list[str] = field(default_factory=list)
    dataset_info: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """True while training or exporting."""
        return self.status in ACTIVE_STATUSES


def validate_name(name: str) -> str:
    """Returns the name when it is safe as a folder and model name, otherwise raises."""
    if not _NAME_PATTERN.match(name):
        raise ValueError(
            "Run name must be 1-64 characters of letters, digits, '.', '_' or '-', "
            "and start with a letter or digit."
        )
    return name


def run_dir(runs_dir: Path, name: str) -> Path:
    """The folder of a run."""
    return runs_dir / validate_name(name=name)


def write_record(run_dir: Path, record: RunRecord) -> None:
    """Writes run.json atomically, so a reader never sees a partial file."""
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / RUN_FILE
    tmp = run_dir / f"{RUN_FILE}.tmp"
    tmp.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    os.replace(tmp, target)


def read_record(run_dir: Path) -> RunRecord | None:
    """Reads run.json of a folder; None when the folder is not a run."""
    path = run_dir / RUN_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunRecord(**data)


def list_records(runs_dir: Path) -> list[RunRecord]:
    """All runs, newest first."""
    if not runs_dir.is_dir():
        return []
    records = []
    for folder in runs_dir.iterdir():
        record = read_record(run_dir=folder)
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda r: r.created_at, reverse=True)


def read_progress(run_dir: Path) -> TrainingProgress:
    """Parses the log of a run; empty progress when there is no log yet."""
    path = run_dir / LOG_FILE
    if not path.is_file():
        return TrainingProgress()
    return log_metrics.parse_log(text=path.read_text(encoding="utf-8", errors="replace"))


def read_log_tail(run_dir: Path, lines: int) -> str:
    """The last lines of the log, or an empty string without a log."""
    path = run_dir / LOG_FILE
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def delete_run(runs_dir: Path, name: str) -> None:
    """Removes a run folder.

    Raises:
        ValueError: If the folder is not a run of runs_dir.
    """
    folder = run_dir(runs_dir=runs_dir, name=name)
    if read_record(run_dir=folder) is None:
        raise ValueError(f"Run '{name}' does not exist.")
    shutil.rmtree(folder)


def now() -> str:
    """The current UTC time as an ISO string with seconds."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
