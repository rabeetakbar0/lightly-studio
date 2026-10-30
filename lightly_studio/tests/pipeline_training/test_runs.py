from pathlib import Path

import pytest

from lightly_studio.pipeline_training import runs
from lightly_studio.pipeline_training.runs import RunRecord


def _record(name: str, created_at: str) -> RunRecord:
    return RunRecord(
        name=name,
        dataset="data1",
        classes=["GateArm"],
        config={"max_epoch": 1},
        status=runs.STATUS_FINISHED,
        created_at=created_at,
    )


def test_validate_name() -> None:
    assert runs.validate_name(name="yolox_m.v5-1") == "yolox_m.v5-1"
    with pytest.raises(ValueError, match="Run name"):
        runs.validate_name(name="../x")


def test_write_and_read_record__round_trip(tmp_path: Path) -> None:
    record = _record("r1", "2026-10-27T10:00:00+00:00")
    record.pid = 42
    runs.write_record(run_dir=tmp_path / "r1", record=record)
    assert runs.read_record(run_dir=tmp_path / "r1") == record
    assert runs.read_record(run_dir=tmp_path / "missing") is None


def test_list_records__newest_first(tmp_path: Path) -> None:
    runs.write_record(run_dir=tmp_path / "old", record=_record("old", "2026-10-26T10:00:00"))
    runs.write_record(run_dir=tmp_path / "new", record=_record("new", "2026-10-27T10:00:00"))
    (tmp_path / "not_a_run").mkdir()
    assert [r.name for r in runs.list_records(runs_dir=tmp_path)] == ["new", "old"]
    assert runs.list_records(runs_dir=tmp_path / "missing") == []


def test_read_log_tail_and_progress(tmp_path: Path) -> None:
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    assert runs.read_log_tail(run_dir=run_dir, lines=5) == ""
    assert runs.read_progress(run_dir=run_dir).epoch == 0
    (run_dir / runs.LOG_FILE).write_text(
        "a\nb\nepoch: 3/10, iter: 1/2, lr: 0, size: 640, ETA: 0:01:00\n"
    )
    assert runs.read_log_tail(run_dir=run_dir, lines=2).splitlines() == [
        "b",
        "epoch: 3/10, iter: 1/2, lr: 0, size: 640, ETA: 0:01:00",
    ]
    assert runs.read_progress(run_dir=run_dir).epoch == 3


def test_delete_run(tmp_path: Path) -> None:
    runs.write_record(run_dir=tmp_path / "r1", record=_record("r1", "2026-10-27T10:00:00"))
    runs.delete_run(runs_dir=tmp_path, name="r1")
    assert not (tmp_path / "r1").exists()
    with pytest.raises(ValueError, match="does not exist"):
        runs.delete_run(runs_dir=tmp_path, name="r1")
