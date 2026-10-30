"""The models folder: ONNX detectors, their checkpoints and their training history.

A model is `<name>.onnx` with `<name>_classes.txt`. A model that can be a baseline for a new
run also has `<name>.pth`, the YOLOX checkpoint. A model trained through the pipeline also has
`<name>.json`, the model card with the dataset, the settings and the scores of its run.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CLASSES_SUFFIX = "_classes.txt"
CHECKPOINT_SUFFIX = ".pth"
CARD_SUFFIX = ".json"

# Depth and width of the stock YOLOX sizes, also used to guess a model's size from its name.
MODEL_PRESETS: dict[str, tuple[float, float]] = {
    "nano": (0.33, 0.25),
    "tiny": (0.33, 0.375),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.0, 1.0),
    "x": (1.33, 1.25),
}
_SIZE_IN_NAME = re.compile(r"yolox[_-](nano|tiny|s|m|l|x)(?![a-z])", re.IGNORECASE)


@dataclass
class ModelCard:
    """How a model was trained; written next to the ONNX file at the end of a run.

    Attributes:
        run_name: The run folder the model came from.
        dataset: The prepared dataset as it was at the start: name, classes, image and box
            counts, splits, annotation source and merged sources.
        baseline: What the run started from, `model:<name>`, `weights:<file>` or None.
        config: The TrainConfig of the run.
        best_ap50_95: Best COCO AP over the run's evaluations, as a fraction.
        evaluations: Per-epoch scores, dicts with epoch, ap50 and ap50_95.
        trained_epochs: Epochs the run reached.
        created_at: When the run started.
        finished_at: When the export finished.
    """

    run_name: str
    dataset: dict[str, Any]
    baseline: str | None
    config: dict[str, Any]
    best_ap50_95: float | None
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    trained_epochs: int = 0
    created_at: str = ""
    finished_at: str = ""


@dataclass
class ModelInfo:
    """One model of the models folder.

    Attributes:
        name: The file stem; also the Detect operator name.
        onnx_path: The ONNX file.
        classes: Class names in output order.
        checkpoint_path: The `.pth` checkpoint, when the model can be a baseline.
        card: The training history, when the model was trained through the pipeline.
        size_bytes: Size of the ONNX file.
        modified_at: ISO timestamp of the ONNX file.
        depth: Depth factor, from the card or guessed from the name; None when unknown.
        width: Width factor, likewise.
    """

    name: str
    onnx_path: Path
    classes: list[str]
    checkpoint_path: Path | None
    card: ModelCard | None
    size_bytes: int
    modified_at: str
    depth: float | None
    width: float | None


def list_models(models_dir: Path) -> list[ModelInfo]:
    """The models in the folder, newest first. ONNX files without a class list are skipped."""
    if not models_dir.is_dir():
        return []
    models = []
    for onnx_path in models_dir.glob("*.onnx"):
        classes_path = onnx_path.with_name(f"{onnx_path.stem}{CLASSES_SUFFIX}")
        if not classes_path.is_file():
            continue
        card = read_card(models_dir=models_dir, name=onnx_path.stem)
        depth, width = _size_of(name=onnx_path.stem, card=card)
        checkpoint = onnx_path.with_suffix(CHECKPOINT_SUFFIX)
        stat = onnx_path.stat()
        models.append(
            ModelInfo(
                name=onnx_path.stem,
                onnx_path=onnx_path,
                classes=_read_lines(classes_path),
                checkpoint_path=checkpoint if checkpoint.is_file() else None,
                card=card,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(
                    timespec="seconds"
                ),
                depth=depth,
                width=width,
            )
        )
    return sorted(models, key=lambda m: m.modified_at, reverse=True)


def checkpoint_path(models_dir: Path, name: str) -> Path:
    """Where the checkpoint of a model lives."""
    return models_dir / f"{name}{CHECKPOINT_SUFFIX}"


def write_card(models_dir: Path, name: str, card: ModelCard) -> Path:
    """Writes the model card next to the ONNX file and returns its path."""
    path = models_dir / f"{name}{CARD_SUFFIX}"
    path.write_text(json.dumps(asdict(card), indent=2), encoding="utf-8")
    return path


def read_card(models_dir: Path, name: str) -> ModelCard | None:
    """The model card, or None when the model has no recorded history."""
    path = models_dir / f"{name}{CARD_SUFFIX}"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelCard(**data)


def size_from_name(name: str) -> tuple[float, float] | None:
    """Depth and width when the name says the YOLOX size, as in yolox_m_slm_v4 or yolox_l.pth."""
    match = _SIZE_IN_NAME.search(name)
    return MODEL_PRESETS[match.group(1).lower()] if match else None


def _size_of(name: str, card: ModelCard | None) -> tuple[float | None, float | None]:
    if card is not None and "depth" in card.config and "width" in card.config:
        return float(card.config["depth"]), float(card.config["width"])
    size = size_from_name(name=name)
    return size if size is not None else (None, None)


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
