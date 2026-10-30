"""Reads progress and validation scores out of a YOLOX training log."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PROGRESS = re.compile(r"epoch: (\d+)/(\d+), iter: (\d+)/(\d+).*?ETA: (\S+)")
_EPOCH_START = re.compile(r"start train epoch(\d+)")
_AP = re.compile(
    r"Average Precision\s+\(AP\) @\[ IoU=(0\.50:0\.95|0\.50)\s+\| area=\s+all "
    r"\| maxDets=100 \] = (-?[\d.]+)"
)


@dataclass
class EpochEvaluation:
    """COCO scores after one epoch, as fractions."""

    epoch: int
    ap50: float | None = None
    ap50_95: float | None = None


@dataclass
class TrainingProgress:
    """What the log says so far."""

    epoch: int = 0
    max_epoch: int = 0
    iteration: int = 0
    max_iteration: int = 0
    eta: str = ""
    evaluations: list[EpochEvaluation] = field(default_factory=list)
    last_line: str = ""

    @property
    def best_ap50_95(self) -> float | None:
        """The best AP over all evaluations, or None before the first one."""
        scores = [e.ap50_95 for e in self.evaluations if e.ap50_95 is not None]
        return max(scores) if scores else None


def parse_log(text: str) -> TrainingProgress:
    """Parses the captured stdout of tools/train.py.

    Progress comes from the periodic "epoch: a/b, iter: c/d ... ETA: x" lines. Scores come
    from the COCO summary that follows "start train epochN"; the IoU=0.50:0.95 and IoU=0.50
    "area=all" lines are kept, the rest of the table is ignored.
    """
    progress = TrainingProgress()
    current_epoch = 0
    evaluations: dict[int, EpochEvaluation] = {}
    for line in text.splitlines():
        if line.strip():
            progress.last_line = line.strip()
        match = _EPOCH_START.search(line)
        if match:
            current_epoch = int(match.group(1))
            continue
        match = _PROGRESS.search(line)
        if match:
            progress.epoch = int(match.group(1))
            progress.max_epoch = int(match.group(2))
            progress.iteration = int(match.group(3))
            progress.max_iteration = int(match.group(4))
            progress.eta = match.group(5)
            continue
        match = _AP.search(line)
        if match and current_epoch > 0:
            evaluation = evaluations.setdefault(current_epoch, EpochEvaluation(epoch=current_epoch))
            value = float(match.group(2))
            if match.group(1) == "0.50":
                evaluation.ap50 = value
            else:
                evaluation.ap50_95 = value
    progress.evaluations = [evaluations[epoch] for epoch in sorted(evaluations)]
    return progress
