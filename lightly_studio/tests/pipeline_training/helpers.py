"""Fixtures shared by the training tests: a fake YOLOX checkout and a prepared dataset."""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

from pytest_mock import MockerFixture

from lightly_studio.dataset import env
from lightly_studio.pipeline_training import runner as runner_module
from lightly_studio.pipeline_training.runner import TrainingRunner

# The stub train.py prints two lines of a real log, writes a checkpoint and exits with
# STUB_EXIT_CODE after sleeping STUB_SLEEP_SECONDS.
_TRAIN_STUB = """
import os, sys, time
args = sys.argv[1:]
run_dir = os.path.dirname(args[args.index("-f") + 1])
print("args:", " ".join(args), flush=True)
print("---> start train epoch1", flush=True)
print("epoch: 1/2, iter: 20/40, gpu mem: 1Mb, lr: 1.0e-03, size: 640, ETA: 0:00:10", flush=True)
print(" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.500", flush=True)
print(" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.800", flush=True)
os.makedirs(os.path.join(run_dir, "train"), exist_ok=True)
with open(os.path.join(run_dir, "train", "best_ckpt.pth"), "wb") as f:
    f.write(b"ckpt")
time.sleep(float(os.environ.get("STUB_SLEEP_SECONDS", "0")))
sys.exit(int(os.environ.get("STUB_EXIT_CODE", "0")))
"""

_EXPORT_STUB = """
import sys
args = sys.argv[1:]
out = args[args.index("--output") + 1]
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as f:
    f.write(b"onnx")
print("exported:", out)
"""


def setup_environment(tmp_path: Path, mocker: MockerFixture) -> TrainingRunner:
    """Points every folder at tmp_path, installs the stubs and returns a fresh runner."""
    yolox_dir = tmp_path / "yolox"
    (yolox_dir / "tools").mkdir(parents=True)
    (yolox_dir / "tools" / "train.py").write_text(textwrap.dedent(_TRAIN_STUB))
    export_script = tmp_path / "export_stub.py"
    export_script.write_text(textwrap.dedent(_EXPORT_STUB))
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "yolox_m.pth").write_bytes(b"w")

    mocker.patch.object(env, "LIGHTLY_STUDIO_RUNS_DIR", tmp_path / "runs")
    mocker.patch.object(env, "LIGHTLY_STUDIO_MODELS_DIR", tmp_path / "models")
    mocker.patch.object(env, "LIGHTLY_STUDIO_WEIGHTS_DIR", weights_dir)
    mocker.patch.object(env, "LIGHTLY_STUDIO_DATASETS_DIR", tmp_path / "datasets")
    mocker.patch.object(env, "LIGHTLY_STUDIO_YOLOX_DIR", yolox_dir)
    mocker.patch.object(env, "LIGHTLY_STUDIO_TRAIN_PYTHON", sys.executable)
    mocker.patch.object(runner_module, "EXPORT_SCRIPT", export_script)
    runner = TrainingRunner()
    mocker.patch.object(runner_module, "training_runner", runner)
    return runner


def write_dataset(tmp_path: Path, name: str, classes: list[str]) -> Path:
    """Writes just the manifest of a prepared dataset; training only reads that."""
    dataset_dir = tmp_path / "datasets" / name
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "dataset.json").write_text(
        json.dumps(
            {
                "name": name,
                "created_at": "2026-10-27T10:00:00+00:00",
                "classes": classes,
                "splits": {"train2017": {"images": 2, "boxes": 2}},
            }
        )
    )
    return dataset_dir


def write_model(tmp_path: Path, name: str, classes: list[str], checkpoint: bool = False) -> Path:
    """Writes a model into the models folder: ONNX, class list and optionally a checkpoint."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / f"{name}.onnx").write_bytes(b"onnx")
    (models_dir / f"{name}_classes.txt").write_text("\n".join(classes) + "\n")
    if checkpoint:
        (models_dir / f"{name}.pth").write_bytes(b"ckpt")
    return models_dir


def wait_until_done(runner: TrainingRunner, name: str, timeout: float = 20.0) -> None:
    """Polls the run until it is no longer active."""
    deadline = time.monotonic() + timeout
    while runner.get_run(name=name).is_active:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Run {name} is still active.")
        time.sleep(0.05)
