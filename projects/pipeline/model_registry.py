"""Finds the detection models in a folder.

A model is an ONNX file with a class list next to it, `<stem>.onnx` and `<stem>_classes.txt`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CLASSES_SUFFIX = "_classes.txt"


@dataclass(frozen=True)
class DetectionModel:
    """One registered model.

    Attributes:
        name: The file stem, used as the operator name and the default annotation layer.
        onnx_path: The ONNX file.
        class_names: Class names in model output order.
    """

    name: str
    onnx_path: Path
    class_names: tuple[str, ...]

    @property
    def default_annotation_source(self) -> str:
        """The annotation layer the model writes to unless the user picks another."""
        return f"{self.name}_prediction"


def discover_models(models_dir: Path) -> list[DetectionModel]:
    """Returns the models in models_dir, sorted by name.

    An ONNX file without a class list, or with an empty one, is skipped with a warning.
    """
    if not models_dir.is_dir():
        logger.warning("Models folder %s does not exist.", models_dir)
        return []
    models: list[DetectionModel] = []
    for onnx_path in sorted(models_dir.glob("*.onnx")):
        classes_path = classes_path_for(onnx_path=onnx_path)
        if not classes_path.is_file():
            logger.warning("Skipping %s: missing %s", onnx_path.name, classes_path.name)
            continue
        class_names = read_class_names(classes_path=classes_path)
        if not class_names:
            logger.warning("Skipping %s: %s is empty", onnx_path.name, classes_path.name)
            continue
        models.append(
            DetectionModel(name=onnx_path.stem, onnx_path=onnx_path, class_names=class_names)
        )
    return models


def classes_path_for(onnx_path: Path) -> Path:
    """The class list that belongs to an ONNX file."""
    return onnx_path.with_name(f"{onnx_path.stem}{CLASSES_SUFFIX}")


def read_class_names(classes_path: Path) -> tuple[str, ...]:
    """Reads one class name per line, skipping blank lines."""
    lines = classes_path.read_text(encoding="utf-8").splitlines()
    return tuple(line.strip() for line in lines if line.strip())
