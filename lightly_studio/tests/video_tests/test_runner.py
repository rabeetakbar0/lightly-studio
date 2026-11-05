from __future__ import annotations

import io
import os
import re
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from lightly_studio.dataset import env
from lightly_studio.video_tests import log_progress, runs
from lightly_studio.video_tests.runner import VideoTestRunner
from tests.video_tests import helpers


def _files(*names: str) -> list[tuple[str, io.BytesIO]]:
    return [(name, io.BytesIO(b"v" * (index + 1))) for index, name in enumerate(names)]


class TestVideoTestRunner:
    def test_environment(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        environment = runner.environment()
        assert environment.ready
        assert environment.uploads_dir == str(tmp_path / "TestApp" / "uploads")
        assert environment.results_dir == str(tmp_path / "TestApp" / "uploads" / "results")

    def test_environment__not_configured(self, tmp_path: Path, mocker: MockerFixture) -> None:
        helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        mocker.patch.object(env, "LIGHTLY_STUDIO_TESTAPP_DIR", tmp_path)
        mocker.patch.object(env, "LIGHTLY_STUDIO_TESTAPP_PYTHON", "/no/such/python")
        environment = VideoTestRunner().environment()
        assert environment.problems == [
            f"{tmp_path} has no App/test_pipeline.py.",
            "TestApp interpreter /no/such/python does not exist.",
        ]
        with pytest.raises(ValueError, match=r"has no App/test_pipeline\.py"):
            VideoTestRunner().start(model="model_a", files=_files("a.mp4"))

    def test_start__finishes(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        record = runner.start(model="model_a", files=_files("a.mp4", "dir/broken.MP4", "a.mp4"))
        assert re.fullmatch(r"\d{8}-\d{6}", record.name)
        assert record.status == runs.STATUS_RUNNING
        uploads = tmp_path / "TestApp" / "uploads"
        input_dir = uploads / record.name
        output_dir = uploads / "results" / record.name
        assert record.input_dir == str(input_dir)
        assert record.output_dir == str(output_dir)
        assert sorted(os.listdir(input_dir)) == ["a (2).mp4", "a.mp4", "broken.MP4"]
        assert [v.file_name for v in record.videos] == ["a.mp4", "broken.MP4", "a (2).mp4"]
        assert [v.size_bytes for v in record.videos] == [1, 2, 3]

        helpers.wait_until_done(runner, record.name)
        record = runner.get_run(name=record.name)
        assert record.status == runs.STATUS_FINISHED
        assert record.exit_code == 0
        assert record.error is None
        a, broken, a2 = record.videos
        assert a.status == log_progress.VIDEO_DONE
        assert a.output_path == str(output_dir / "a_processed.mp4")
        assert (output_dir / "a_processed.mp4").read_bytes() == b"mp4"
        assert a.summary == {"GatearmDetected": "True"}
        assert a.benchmark == {"fps": "100.0"}
        assert broken.status == log_progress.VIDEO_FAILED
        assert a2.output_path == str(output_dir / "a (2)_processed.mp4")
        log = runner.read_log(name=record.name, lines=100)
        assert f"Model : {tmp_path / 'models' / 'model_a.onnx'}" in log
        assert f"cwd: {tmp_path / 'TestApp'}" in log
        assert runs.read_record(run_dir=output_dir) == record

    def test_start__every_video_fails(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        record = runner.start(model="model_a", files=_files("broken.mp4"))
        helpers.wait_until_done(runner, record.name)
        record = runner.get_run(name=record.name)
        assert record.status == runs.STATUS_FAILED
        assert record.error == "No video was processed; see the log."

    def test_start__exit_code(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        mocker.patch.dict(os.environ, {"STUB_EXIT_CODE": "3"})
        record = runner.start(model="model_a", files=_files("a.mp4"))
        helpers.wait_until_done(runner, record.name)
        record = runner.get_run(name=record.name)
        assert record.status == runs.STATUS_FAILED
        assert record.error == "The pipeline exited with code 3; see the log."

    def test_start__rejects(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        with pytest.raises(ValueError, match="Model 'nope' is not in the models folder"):
            runner.start(model="nope", files=_files("a.mp4"))
        with pytest.raises(ValueError, match="Add at least one video"):
            runner.start(model="model_a", files=[])
        with pytest.raises(ValueError, match=r"Not a video .*: notes.txt"):
            runner.start(model="model_a", files=_files("a.mp4", "notes.txt"))
        assert not (tmp_path / "TestApp" / "uploads").exists()

    def test_start__one_at_a_time_and_stop(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        mocker.patch.dict(os.environ, {"STUB_SLEEP_SECONDS": "30"})
        record = runner.start(model="model_a", files=_files("a.mp4"))
        with pytest.raises(ValueError, match="already running"):
            runner.start(model="model_a", files=_files("b.mp4"))

        stopped = runner.stop(name=record.name)
        assert stopped.status == runs.STATUS_STOPPED
        helpers.wait_until_done(runner, record.name)
        assert runner.get_run(name=record.name).status == runs.STATUS_STOPPED
        with pytest.raises(ValueError, match="is not running"):
            runner.stop(name=record.name)
        # A new run gets its own folders.
        mocker.patch.dict(os.environ, {"STUB_SLEEP_SECONDS": "0"})
        second = runner.start(model="model_a", files=_files("b.mp4"))
        assert second.name != record.name
        helpers.wait_until_done(runner, second.name)
        assert [r.name for r in runner.list_runs()] == [second.name, record.name]

    def test_reconcile(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        output_dir = tmp_path / "TestApp" / "uploads" / "results" / "20261102-170000"
        runs.write_record(
            run_dir=output_dir,
            record=runs.VideoTestRun(
                name="20261102-170000",
                model="model_a",
                model_path="m.onnx",
                input_dir="in",
                output_dir=str(output_dir),
                status=runs.STATUS_RUNNING,
                created_at="2026-11-02T17:00:00+00:00",
            ),
        )
        record = runner.get_run(name="20261102-170000")
        assert record.status == runs.STATUS_FAILED
        assert record.error == "The backend was restarted while the video test was running."

    def test_get_run__unknown(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        with pytest.raises(ValueError, match="does not exist"):
            runner.get_run(name="nope")
