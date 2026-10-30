from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from lightly_studio.dataset import env
from lightly_studio.pipeline_training import model_catalog, runs
from lightly_studio.pipeline_training.config import TrainConfig
from lightly_studio.pipeline_training.runner import TrainingRunner
from tests.pipeline_training import helpers


class TestTrainingRunner:
    def test_environment(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        environment = runner.environment()
        assert environment.ready
        assert environment.weights == ["yolox_m.pth"]

    def test_environment__not_configured(self, tmp_path: Path, mocker: MockerFixture) -> None:
        helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        mocker.patch.object(env, "LIGHTLY_STUDIO_YOLOX_DIR", None)
        mocker.patch.object(env, "LIGHTLY_STUDIO_TRAIN_PYTHON", "/no/such/python")
        environment = TrainingRunner().environment()
        assert not environment.ready
        assert environment.problems == [
            "LIGHTLY_STUDIO_YOLOX_DIR is not set.",
            "Training interpreter /no/such/python does not exist.",
        ]

    def test_start__finishes_and_exports(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        exported: list[Path] = []
        runner.on_model_exported = exported.append
        dataset_dir = helpers.write_dataset(tmp_path, "data1", ["GateArm", "LedLight"])

        record = runner.start(
            name="run1", dataset="data1", config=TrainConfig(max_epoch=2, cache="ram")
        )
        assert record.status == runs.STATUS_RUNNING
        assert record.classes == ["GateArm", "LedLight"]
        run_dir = tmp_path / "runs" / "run1"
        exp_source = (run_dir / runs.EXP_FILE).read_text()
        assert f"self.data_dir = '{dataset_dir}'" in exp_source
        assert "self.num_classes = 2" in exp_source
        assert record.command[-5:] == [
            "--fp16",
            "--cache",
            "ram",
            "-c",
            str(tmp_path / "weights" / "yolox_m.pth"),
        ]

        helpers.wait_until_done(runner, "run1")
        record = runner.get_run(name="run1")
        assert record.status == runs.STATUS_FINISHED
        assert record.exit_code == 0
        assert record.onnx_path == str(tmp_path / "models" / "run1.onnx")
        assert (tmp_path / "models" / "run1.onnx").read_bytes() == b"onnx"
        assert (tmp_path / "models" / "run1_classes.txt").read_text() == "GateArm\nLedLight\n"
        assert exported == [tmp_path / "models" / "run1.onnx"]
        assert (tmp_path / "models" / "run1.pth").read_bytes() == b"ckpt"
        card = model_catalog.read_card(models_dir=tmp_path / "models", name="run1")
        assert card is not None
        assert card.dataset["name"] == "data1"
        assert card.dataset["images"] == 2
        assert card.baseline == "weights:yolox_m.pth"
        assert card.config["max_epoch"] == 2
        assert card.best_ap50_95 == 0.5
        assert card.evaluations == [{"epoch": 1, "ap50": 0.8, "ap50_95": 0.5}]
        assert card.trained_epochs == 1
        progress = runs.read_progress(run_dir=run_dir)
        assert progress.best_ap50_95 == 0.5
        assert "=== ONNX export ===" in runs.read_log_tail(run_dir=run_dir, lines=10)

    def test_start__without_export(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        helpers.write_dataset(tmp_path, "data1", ["GateArm"])
        runner.start(
            name="run1",
            dataset="data1",
            config=TrainConfig(export_onnx=False, baseline=None, fp16=False),
        )
        helpers.wait_until_done(runner, "run1")
        record = runner.get_run(name="run1")
        assert record.status == runs.STATUS_FINISHED
        assert record.onnx_path is None
        assert "-c" not in record.command
        assert "--fp16" not in record.command
        assert not (tmp_path / "models").exists()

    def test_start__from_a_model_baseline(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        helpers.write_dataset(tmp_path, "data1", ["GateArm"])
        helpers.write_model(tmp_path, "yolox_m_slm_v4", ["GateArm", "LedLight"], checkpoint=True)
        record = runner.start(
            name="run2",
            dataset="data1",
            config=TrainConfig(baseline="model:yolox_m_slm_v4", export_onnx=False),
        )
        assert record.command[-2:] == ["-c", str(tmp_path / "models" / "yolox_m_slm_v4.pth")]
        helpers.wait_until_done(runner, "run2")
        assert runner.get_run(name="run2").status == runs.STATUS_FINISHED

    def test_start__failed_process(
        self, tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        helpers.write_dataset(tmp_path, "data1", ["GateArm"])
        monkeypatch.setenv("STUB_EXIT_CODE", "3")
        runner.start(name="run1", dataset="data1", config=TrainConfig())
        helpers.wait_until_done(runner, "run1")
        record = runner.get_run(name="run1")
        assert record.status == runs.STATUS_FAILED
        assert record.exit_code == 3
        assert record.error == "Training exited with code 3; see the log."

    def test_start__rejects_bad_input(self, tmp_path: Path, mocker: MockerFixture) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        helpers.write_dataset(tmp_path, "data1", ["GateArm"])
        helpers.write_dataset(tmp_path, "empty", [])
        with pytest.raises(ValueError, match="does not exist"):
            runner.start(name="r", dataset="missing", config=TrainConfig())
        with pytest.raises(ValueError, match="no classes"):
            runner.start(name="r", dataset="empty", config=TrainConfig())
        with pytest.raises(ValueError, match="not in the weights folder"):
            runner.start(name="r", dataset="data1", config=TrainConfig(baseline="weights:nope.pth"))
        with pytest.raises(ValueError, match="has no checkpoint"):
            runner.start(name="r", dataset="data1", config=TrainConfig(baseline="model:nope"))
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "taken.onnx").write_bytes(b"")
        with pytest.raises(ValueError, match="exists already in the models folder"):
            runner.start(name="taken", dataset="data1", config=TrainConfig())
        mocker.patch.object(env, "LIGHTLY_STUDIO_YOLOX_DIR", None)
        with pytest.raises(ValueError, match="LIGHTLY_STUDIO_YOLOX_DIR"):
            runner.start(name="r", dataset="data1", config=TrainConfig())

    def test_stop(
        self, tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        helpers.write_dataset(tmp_path, "data1", ["GateArm"])
        monkeypatch.setenv("STUB_SLEEP_SECONDS", "30")
        runner.start(name="run1", dataset="data1", config=TrainConfig())
        with pytest.raises(ValueError, match="still active"):
            runner.start(name="run2", dataset="data1", config=TrainConfig())
        with pytest.raises(ValueError, match="is active"):
            runner.delete(name="run1")

        assert runner.stop(name="run1").status == runs.STATUS_STOPPED
        helpers.wait_until_done(runner, "run1")
        record = runner.get_run(name="run1")
        assert record.status == runs.STATUS_STOPPED
        assert record.onnx_path is None
        with pytest.raises(ValueError, match="not running"):
            runner.stop(name="run1")

        runner.delete(name="run1")
        assert runner.list_runs() == []

    def test_list_runs__reconciles_runs_of_an_earlier_process(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
        stale = runs.RunRecord(
            name="stale",
            dataset="data1",
            classes=["GateArm"],
            config={},
            status=runs.STATUS_RUNNING,
            created_at="2026-10-27T10:00:00+00:00",
            pid=1,
        )
        runs.write_record(run_dir=tmp_path / "runs" / "stale", record=stale)
        records = TrainingRunner().list_runs()
        assert records[0].status == runs.STATUS_FAILED
        assert records[0].error == "The backend was restarted while the run was active."
