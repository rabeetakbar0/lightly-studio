"""Starts, watches and stops YOLOX training processes.

A run is a subprocess of YOLOX's tools/train.py in its own run folder, started with the
training interpreter (LIGHTLY_STUDIO_TRAIN_PYTHON) against the YOLOX checkout
(LIGHTLY_STUDIO_YOLOX_DIR). Its output goes to log.txt in the run folder. A thread per run
waits for the process, then exports ONNX into the models folder when asked to.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightly_studio.dataset import env
from lightly_studio.pipeline_datasets import coco_dataset
from lightly_studio.pipeline_datasets.coco_dataset import TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT
from lightly_studio.pipeline_training import exp_writer, log_metrics, model_catalog, runs
from lightly_studio.pipeline_training.config import BASELINE_MODEL, TrainConfig
from lightly_studio.pipeline_training.model_catalog import ModelCard, ModelInfo
from lightly_studio.pipeline_training.runs import RunRecord

logger = logging.getLogger(__name__)

EXPORT_SCRIPT = Path(__file__).with_name("export_onnx_script.py")
CLASSES_SUFFIX = "_classes.txt"
STOP_TIMEOUT_SECONDS = 30


@dataclass
class TrainingEnvironment:
    """Whether training can start on this machine, and with what.

    Attributes:
        yolox_dir: The YOLOX checkout, or None when not configured.
        python: The training interpreter, or None when not configured.
        weights: Pretrained checkpoint files in the weights folder.
        problems: What is missing; training can start when this is empty.
    """

    yolox_dir: str | None
    python: str | None
    weights: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True when nothing is missing."""
        return not self.problems


class TrainingRunner:
    """The runs of this backend process.

    Attributes:
        on_model_exported: Called with the ONNX path after a successful export, so the
            launcher can register the model as a Detect operator without a restart.
    """

    def __init__(self) -> None:
        """Starts with no runs; existing run folders are picked up on first use."""
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._stop_requested: set[str] = set()
        self._lock = threading.Lock()
        self._reconciled = False
        self.on_model_exported: Callable[[Path], None] | None = None

    def environment(self) -> TrainingEnvironment:
        """Checks the configured interpreter, YOLOX checkout and weights folder."""
        yolox_dir = env.LIGHTLY_STUDIO_YOLOX_DIR
        python = env.LIGHTLY_STUDIO_TRAIN_PYTHON
        problems = []
        if yolox_dir is None:
            problems.append("LIGHTLY_STUDIO_YOLOX_DIR is not set.")
        elif not (yolox_dir / "tools" / "train.py").is_file():
            problems.append(f"{yolox_dir} has no tools/train.py.")
        if python is None:
            problems.append("LIGHTLY_STUDIO_TRAIN_PYTHON is not set.")
        elif shutil.which(python) is None and not Path(python).is_file():
            problems.append(f"Training interpreter {python} does not exist.")
        weights_dir = env.LIGHTLY_STUDIO_WEIGHTS_DIR
        weights = sorted(p.name for p in weights_dir.glob("*.pth")) if weights_dir.is_dir() else []
        return TrainingEnvironment(
            yolox_dir=str(yolox_dir) if yolox_dir else None,
            python=python,
            weights=weights,
            problems=problems,
        )

    def list_runs(self) -> list[RunRecord]:
        """All runs in the runs folder, newest first."""
        self._reconcile()
        return runs.list_records(runs_dir=env.LIGHTLY_STUDIO_RUNS_DIR)

    def get_run(self, name: str) -> RunRecord:
        """One run.

        Raises:
            ValueError: If there is no such run.
        """
        self._reconcile()
        record = runs.read_record(run_dir=self._run_dir(name=name))
        if record is None:
            raise ValueError(f"Run '{name}' does not exist.")
        return record

    def start(self, name: str, dataset: str, config: TrainConfig) -> RunRecord:
        """Creates the run folder, writes the exp file and launches training.

        Raises:
            ValueError: If the environment is not ready, the dataset or weights do not exist,
                the name is taken, or another run is active.
        """
        self._reconcile()
        environment = self.environment()
        if not environment.ready:
            raise ValueError(" ".join(environment.problems))
        assert environment.yolox_dir is not None
        assert environment.python is not None
        run_dir = self._run_dir(name=name)
        if run_dir.exists():
            raise ValueError(f"Run '{name}' exists already.")
        onnx_target = env.LIGHTLY_STUDIO_MODELS_DIR / f"{name}.onnx"
        if config.export_onnx and onnx_target.exists():
            raise ValueError(f"Model '{onnx_target.name}' exists already in the models folder.")
        active = [
            r.name for r in runs.list_records(runs_dir=env.LIGHTLY_STUDIO_RUNS_DIR) if r.is_active
        ]
        if active:
            raise ValueError(f"Run '{active[0]}' is still active; one run at a time.")

        coco_dataset.validate_name(name=dataset)
        dataset_dir = env.LIGHTLY_STUDIO_DATASETS_DIR / dataset
        summary = coco_dataset.read_summary(dataset_dir=dataset_dir)
        if summary is None:
            raise ValueError(f"Dataset '{dataset}' does not exist.")
        if not summary.classes:
            raise ValueError(f"Dataset '{dataset}' has no classes.")
        weights = _resolve_baseline(config=config)

        run_dir.mkdir(parents=True)
        (run_dir / runs.EXP_FILE).write_text(
            exp_writer.render_exp(
                config=config,
                data_dir=dataset_dir,
                num_classes=len(summary.classes),
                output_dir=run_dir,
            ),
            encoding="utf-8",
        )
        command = _train_command(
            python=environment.python,
            yolox_dir=Path(environment.yolox_dir),
            run_dir=run_dir,
            config=config,
            weights=weights,
        )
        record = RunRecord(
            name=name,
            dataset=dataset,
            classes=list(summary.classes),
            dataset_info=_dataset_info(summary=summary),
            config=config.model_dump(),
            status=runs.STATUS_RUNNING,
            created_at=runs.now(),
            command=command,
        )
        return self._launch(record=record, run_dir=run_dir, python=environment.python)

    def start_test(self, model: str, dataset: str, batch_size: int, fp16: bool) -> RunRecord:
        """Evaluates a model's checkpoint on the test split of a dataset.

        Runs YOLOX's tools/eval.py, the same evaluator as during training, in a run folder
        named after the model and the dataset. The result is written to the model card.

        Raises:
            ValueError: If the environment is not ready, the model has no checkpoint, the
                dataset has no test split, the classes differ, or another run is active.
        """
        self._reconcile()
        environment = self.environment()
        if not environment.ready:
            raise ValueError(" ".join(environment.problems))
        assert environment.yolox_dir is not None
        assert environment.python is not None
        info = self._model(name=model)
        if info.checkpoint_path is None:
            raise ValueError(f"Model '{model}' has no checkpoint in the models folder.")
        coco_dataset.validate_name(name=dataset)
        dataset_dir = env.LIGHTLY_STUDIO_DATASETS_DIR / dataset
        summary = coco_dataset.read_summary(dataset_dir=dataset_dir)
        if summary is None:
            raise ValueError(f"Dataset '{dataset}' does not exist.")
        if TEST_SPLIT not in summary.splits:
            raise ValueError(f"Dataset '{dataset}' has no test split.")
        if list(summary.classes) != list(info.classes):
            raise ValueError(
                f"Dataset classes {summary.classes} differ from the model's {info.classes}."
            )
        active = [
            r.name for r in runs.list_records(runs_dir=env.LIGHTLY_STUDIO_RUNS_DIR) if r.is_active
        ]
        if active:
            raise ValueError(f"Run '{active[0]}' is still active; one run at a time.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{model}__test__{dataset}__{stamp}"
        run_dir = self._run_dir(name=name)
        if run_dir.exists():
            raise ValueError(f"Run '{name}' exists already.")

        config = _model_config(info=info)
        config.batch_size = batch_size
        config.fp16 = fp16
        run_dir.mkdir(parents=True)
        (run_dir / runs.EXP_FILE).write_text(
            exp_writer.render_eval_exp(
                config=config,
                data_dir=dataset_dir,
                num_classes=len(summary.classes),
                output_dir=run_dir,
            ),
            encoding="utf-8",
        )
        command = [
            environment.python,
            "-u",
            str(Path(environment.yolox_dir) / "tools" / "eval.py"),
            "-f",
            str(run_dir / runs.EXP_FILE),
            "-c",
            str(info.checkpoint_path),
            "-d",
            "1",
            "-b",
            str(batch_size),
        ]
        if fp16:
            command.append("--fp16")
        record = RunRecord(
            name=name,
            dataset=dataset,
            classes=list(summary.classes),
            dataset_info=_dataset_info(summary=summary),
            config=config.model_dump(),
            status=runs.STATUS_RUNNING,
            created_at=runs.now(),
            command=command,
            kind=runs.KIND_TEST,
            model=model,
        )
        return self._launch(record=record, run_dir=run_dir, python=environment.python)

    def _launch(self, record: RunRecord, run_dir: Path, python: str) -> RunRecord:
        log = (run_dir / runs.LOG_FILE).open("ab")
        try:
            process = subprocess.Popen(
                record.command, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT
            )
        finally:
            log.close()
        record.pid = process.pid
        with self._lock:
            self._processes[record.name] = process
        runs.write_record(run_dir=run_dir, record=record)
        threading.Thread(
            target=self._supervise, args=(record, process, python), daemon=True
        ).start()
        return record

    def _model(self, name: str) -> ModelInfo:
        for info in model_catalog.list_models(models_dir=env.LIGHTLY_STUDIO_MODELS_DIR):
            if info.name == name:
                return info
        raise ValueError(f"Model '{name}' is not in the models folder.")

    def stop(self, name: str) -> RunRecord:
        """Terminates the training process; the run ends as 'stopped'.

        Raises:
            ValueError: If there is no such run or it is not active.
        """
        record = self.get_run(name=name)
        with self._lock:
            process = self._processes.get(name)
            if process is None or process.poll() is not None:
                raise ValueError(f"Run '{name}' is not running.")
            self._stop_requested.add(name)
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
        record.status = runs.STATUS_STOPPED
        return record

    def delete(self, name: str) -> None:
        """Removes a run folder. The exported model, if any, stays in the models folder.

        Raises:
            ValueError: If the run is active or does not exist.
        """
        record = self.get_run(name=name)
        if record.is_active:
            raise ValueError(f"Run '{name}' is active; stop it first.")
        runs.delete_run(runs_dir=env.LIGHTLY_STUDIO_RUNS_DIR, name=name)

    def _supervise(self, record: RunRecord, process: subprocess.Popen[bytes], python: str) -> None:
        run_dir = self._run_dir(name=record.name)
        record.exit_code = process.wait()
        with self._lock:
            self._processes.pop(record.name, None)
            stopped = record.name in self._stop_requested
            self._stop_requested.discard(record.name)
        if stopped:
            record.status = runs.STATUS_STOPPED
        elif record.exit_code != 0:
            record.status = runs.STATUS_FAILED
            what = "Testing" if record.kind == runs.KIND_TEST else "Training"
            record.error = f"{what} exited with code {record.exit_code}; see the log."
        elif record.kind == runs.KIND_TEST:
            self._record_test(record=record, run_dir=run_dir)
        elif record.config.get("export_onnx"):
            record.status = runs.STATUS_EXPORTING
            runs.write_record(run_dir=run_dir, record=record)
            self._export(record=record, run_dir=run_dir, python=python)
        else:
            record.status = runs.STATUS_FINISHED
        record.finished_at = runs.now()
        runs.write_record(run_dir=run_dir, record=record)

    def _export(self, record: RunRecord, run_dir: Path, python: str) -> None:
        checkpoints = run_dir / exp_writer.TRAIN_OUTPUT_NAME
        ckpt = checkpoints / "best_ckpt.pth"
        if not ckpt.is_file():
            ckpt = checkpoints / "latest_ckpt.pth"
        if not ckpt.is_file():
            record.status = runs.STATUS_FAILED
            record.error = "Training wrote no checkpoint, nothing to export."
            return
        export_dir = run_dir / runs.EXPORT_DIR
        onnx_path = export_dir / f"{record.name}.onnx"
        command = [
            python,
            str(EXPORT_SCRIPT),
            "--exp-file",
            str(run_dir / runs.EXP_FILE),
            "--ckpt",
            str(ckpt),
            "--output",
            str(onnx_path),
            "--opset",
            str(record.config.get("onnx_opset", 11)),
        ]
        with (run_dir / runs.LOG_FILE).open("ab") as log:
            log.write(b"\n=== ONNX export ===\n")
            log.flush()
            code = subprocess.call(command, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT)
        if code != 0 or not onnx_path.is_file():
            record.status = runs.STATUS_FAILED
            record.error = f"ONNX export exited with code {code}; see the log."
            return
        models_dir = env.LIGHTLY_STUDIO_MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / onnx_path.name
        shutil.copy2(onnx_path, target)
        classes_text = "\n".join(record.classes) + "\n"
        (export_dir / f"{record.name}{CLASSES_SUFFIX}").write_text(classes_text, encoding="utf-8")
        (models_dir / f"{record.name}{CLASSES_SUFFIX}").write_text(classes_text, encoding="utf-8")
        record.onnx_path = str(target)
        _place_checkpoint(
            src=ckpt, dst=model_catalog.checkpoint_path(models_dir=models_dir, name=record.name)
        )
        progress = runs.read_progress(run_dir=run_dir)
        model_catalog.write_card(
            models_dir=models_dir,
            name=record.name,
            card=ModelCard(
                run_name=record.name,
                dataset=record.dataset_info or {"name": record.dataset, "classes": record.classes},
                baseline=record.config.get("baseline"),
                config=record.config,
                best_ap50_95=progress.best_ap50_95,
                evaluations=[
                    {"epoch": e.epoch, "ap50": e.ap50, "ap50_95": e.ap50_95}
                    for e in progress.evaluations
                ],
                trained_epochs=progress.epoch,
                created_at=record.created_at,
                finished_at=runs.now(),
            ),
        )
        record.status = runs.STATUS_FINISHED
        if self.on_model_exported is not None:
            try:
                self.on_model_exported(target)
            except Exception:
                logger.exception("Registering the exported model %s failed.", target)

    def _record_test(self, record: RunRecord, run_dir: Path) -> None:
        """Reads the scores from the log and appends them to the model card."""
        assert record.model is not None
        result = log_metrics.parse_eval(text=runs.read_log_tail(run_dir=run_dir, lines=10_000))
        if result.ap50_95 is None:
            record.status = runs.STATUS_FAILED
            record.error = "The evaluation printed no COCO summary; see the log."
            return
        models_dir = env.LIGHTLY_STUDIO_MODELS_DIR
        card = model_catalog.read_card(models_dir=models_dir, name=record.model)
        if card is None:
            card = model_catalog.tests_only_card()
        test_images = int(
            record.dataset_info.get("splits", {}).get(TEST_SPLIT, {}).get("images", 0)
        )
        record.test_result = {
            "dataset": record.dataset,
            "images": test_images,
            "ap50_95": result.ap50_95,
            "ap50": result.ap50,
            "per_class_ap": result.per_class_ap,
            "overlap_images": _training_overlap(card=card, test_dataset=record.dataset),
            "run_name": record.name,
            "created_at": record.created_at,
        }
        card.tests.append(record.test_result)
        model_catalog.write_card(models_dir=models_dir, name=record.model, card=card)
        record.status = runs.STATUS_FINISHED

    def _reconcile(self) -> None:
        """Marks runs left active by an earlier backend process as failed, once."""
        with self._lock:
            if self._reconciled:
                return
            self._reconciled = True
        for record in runs.list_records(runs_dir=env.LIGHTLY_STUDIO_RUNS_DIR):
            if record.is_active and record.name not in self._processes:
                record.status = runs.STATUS_FAILED
                record.error = "The backend was restarted while the run was active."
                record.finished_at = runs.now()
                runs.write_record(run_dir=self._run_dir(name=record.name), record=record)

    def _run_dir(self, name: str) -> Path:
        return runs.run_dir(runs_dir=env.LIGHTLY_STUDIO_RUNS_DIR, name=name)


def _resolve_baseline(config: TrainConfig) -> Path | None:
    """The checkpoint file to start from, or None for random initialisation.

    Raises:
        ValueError: If the model has no checkpoint or the weights file does not exist.
    """
    parts = config.baseline_parts()
    if parts is None:
        return None
    kind, name = parts
    if kind == BASELINE_MODEL:
        path = model_catalog.checkpoint_path(models_dir=env.LIGHTLY_STUDIO_MODELS_DIR, name=name)
        if not path.is_file():
            raise ValueError(
                f"Model '{name}' has no checkpoint ({path.name}) in the models folder, "
                "so it cannot be a baseline."
            )
        return path
    path = env.LIGHTLY_STUDIO_WEIGHTS_DIR / name
    if not path.is_file():
        raise ValueError(f"Weights '{name}' are not in the weights folder.")
    return path


def _dataset_info(summary: coco_dataset.DatasetSummary) -> dict[str, Any]:
    return {
        "name": summary.name,
        "classes": list(summary.classes),
        "images": summary.images,
        "boxes": summary.boxes,
        "splits": {
            split: {"images": counts.images, "boxes": counts.boxes}
            for split, counts in summary.splits.items()
        },
        "annotation_source": summary.annotation_source,
        "sources": list(summary.sources),
    }


def _model_config(info: ModelInfo) -> TrainConfig:
    """The model's training settings, or defaults with its size for a model without a card."""
    if info.card is not None and info.card.trained:
        return TrainConfig.model_validate(info.card.config)
    config = TrainConfig()
    if info.depth is not None and info.width is not None:
        config.depth = info.depth
        config.width = info.width
    return config


def _training_overlap(card: ModelCard, test_dataset: str) -> int | None:
    """How many test images were in the model's training data; None when unknown.

    Images are matched by Studio sample id, or by file name when an entry has none. The
    check needs the model's training dataset folder; a model without a recorded dataset,
    or whose dataset was deleted, gives None.
    """
    if not card.trained:
        return None
    training_dir = env.LIGHTLY_STUDIO_DATASETS_DIR / str(card.dataset.get("name", ""))
    if not (training_dir / "annotations").is_dir():
        return None
    seen = _image_keys(dataset_dir=training_dir, splits=(TRAIN_SPLIT, VAL_SPLIT))
    test_keys = _image_keys(
        dataset_dir=env.LIGHTLY_STUDIO_DATASETS_DIR / test_dataset, splits=(TEST_SPLIT,)
    )
    return len(test_keys & seen)


def _image_keys(dataset_dir: Path, splits: tuple[str, ...]) -> set[str]:
    keys: set[str] = set()
    for split in splits:
        path = dataset_dir / "annotations" / f"instances_{split}.json"
        if not path.is_file():
            continue
        for image in json.loads(path.read_text(encoding="utf-8")).get("images", []):
            keys.add(str(image.get("sample_id") or image["file_name"]))
    return keys


def _place_checkpoint(src: Path, dst: Path) -> None:
    """Hard-links the checkpoint into the models folder, copying when linking is impossible."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _train_command(
    python: str, yolox_dir: Path, run_dir: Path, config: TrainConfig, weights: Path | None
) -> list[str]:
    command = [
        python,
        "-u",
        str(yolox_dir / "tools" / "train.py"),
        "-f",
        str(run_dir / runs.EXP_FILE),
        "-d",
        str(config.devices),
        "-b",
        str(config.batch_size),
    ]
    if config.fp16:
        command.append("--fp16")
    if config.occupy:
        command.append("-o")
    if config.cache != "none":
        command.extend(["--cache", config.cache])
    if weights is not None:
        command.extend(["-c", str(weights)])
    return command


# The backend's single runner; the launcher sets its export hook.
training_runner = TrainingRunner()
