"""Starts LightlyStudio on an empty, persistent dataset, with the models folder as plugins.

Nothing is loaded up front. Images and videos are added by dropping them onto the grid in
the GUI. The database file lives next to this script, so everything added survives restarts.

Run from the backend folder, through `python`, so uv uses the backend member's environment
(a bare script path makes uv pick the workspace root and pull the CUDA torch build):

    cd lightly_studio && make start-pipeline          # keep what is there, append new drops
    cd lightly_studio && make start-pipeline-clean    # wipe the database and uploads first
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Environment settings are read when lightly_studio is imported, so they come first.
os.environ.setdefault("LIGHTLY_STUDIO_UPLOAD_DIR", str(PROJECT_DIR / "uploads"))
os.environ.setdefault("LIGHTLY_STUDIO_DATASETS_DIR", str(PROJECT_DIR / "datasets"))
os.environ.setdefault("LIGHTLY_STUDIO_RUNS_DIR", str(PROJECT_DIR / "runs"))
os.environ.setdefault("LIGHTLY_STUDIO_WEIGHTS_DIR", str(PROJECT_DIR / "weights"))
os.environ.setdefault(
    "LIGHTLY_STUDIO_MODELS_DIR", os.environ.get("PIPELINE_MODELS_DIR", str(PROJECT_DIR / "models"))
)

import lightly_studio as ls
from lightly_studio.database import db_manager
from lightly_studio.pipeline_training.runner import training_runner
from lightly_studio.plugins.operator_registry import operator_registry

sys.path.insert(0, str(PROJECT_DIR))
from pipeline import model_registry
from pipeline.detect_operator import DetectOperator

DATASET_NAME = os.environ.get("STUDIO_DATASET_NAME", "pipeline")
DB_FILE = Path(os.environ.get("STUDIO_DB_FILE", PROJECT_DIR / "lightly_studio_pipeline.db"))
MODELS_DIR = Path(os.environ["LIGHTLY_STUDIO_MODELS_DIR"])
UPLOAD_DIR = Path(os.environ["LIGHTLY_STUDIO_UPLOAD_DIR"])


def register_exported_model(onnx_path: Path) -> None:
    """Makes a model that a training run just exported available as a Detect operator."""
    class_names = model_registry.read_class_names(
        classes_path=model_registry.classes_path_for(onnx_path=onnx_path)
    )
    model = model_registry.DetectionModel(
        name=onnx_path.stem, onnx_path=onnx_path, class_names=class_names
    )
    operator_registry.register(operator=DetectOperator(model=model))
    print(f"Registered detection model {model.name} from the training run.")


def reset_slate() -> None:
    """Deletes the database and every uploaded file, so the next start is empty."""
    for path in (DB_FILE, DB_FILE.with_name(DB_FILE.name + ".wal")):
        if path.exists():
            path.unlink()
            print(f"Removed {path}")
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
        print(f"Removed {UPLOAD_DIR}")


if "--reset" in sys.argv[1:] or os.environ.get("STUDIO_RESET") == "1":
    reset_slate()

db_manager.connect(db_file=str(DB_FILE))
dataset = ls.ImageDataset.load_or_create(name=DATASET_NAME)
print(f"Dataset '{DATASET_NAME}' ready. Database: {DB_FILE}")

# Every <name>.onnx with a <name>_classes.txt in models/ becomes a "Detect: <name>" operator.
models = model_registry.discover_models(models_dir=MODELS_DIR)
for model in models:
    operator_registry.register(operator=DetectOperator(model=model))
print(
    f"Registered {len(models)} detection models from {MODELS_DIR}: "
    f"{', '.join(m.name for m in models) or 'none'}"
)

# Training runs (GUI: Train panel) need LIGHTLY_STUDIO_YOLOX_DIR and
# LIGHTLY_STUDIO_TRAIN_PYTHON in lightly_studio/.env; a finished run registers its model here.
training_runner.on_model_exported = register_exported_model

ls.start_gui(host="127.0.0.1", allow_empty=True)
