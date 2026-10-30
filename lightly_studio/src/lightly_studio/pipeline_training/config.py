"""The training settings a user can change, with the values of the v4 gate-arm recipe."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

STRIDE = 32
BASELINE_MODEL = "model"
BASELINE_WEIGHTS = "weights"
_BASELINE = re.compile(r"^(model|weights):([A-Za-z0-9][A-Za-z0-9._-]{0,127})$")


class TrainConfig(BaseModel):
    """Everything that goes into the generated exp file and the train command.

    The defaults are the v4 YOLOX-M recipe (slm_v4_yolox_m.py). Field names follow the YOLOX
    exp attributes so the generated file is easy to compare with a hand-written one.
    """

    # Model.
    depth: float = Field(0.67, gt=0, description="Depth factor; 0.67 is YOLOX-M.")
    width: float = Field(0.75, gt=0, description="Width factor; 0.75 is YOLOX-M.")
    act: Literal["silu", "relu", "lrelu"] = "silu"
    input_size: int = Field(640, ge=STRIDE, multiple_of=STRIDE)
    test_size: int = Field(640, ge=STRIDE, multiple_of=STRIDE)
    multiscale_range: int = Field(5, ge=0)
    max_labels: int = Field(50, ge=1)

    # Schedule.
    max_epoch: int = Field(150, ge=1)
    warmup_epochs: int = Field(5, ge=0)
    no_aug_epochs: int = Field(30, ge=0)
    basic_lr_per_img: float = Field(0.01 / 64.0, gt=0)
    warmup_lr: float = Field(0.0, ge=0)
    min_lr_ratio: float = Field(0.05, ge=0, le=1)
    scheduler: Literal["yoloxwarmcos", "warmcos", "cos", "yoloxsemiwarmcos"] = "yoloxwarmcos"
    ema: bool = True
    weight_decay: float = Field(5e-4, ge=0)
    momentum: float = Field(0.9, ge=0, le=1)
    seed: int | None = Field(None, ge=0)

    # Augmentation.
    mosaic_prob: float = Field(1.0, ge=0, le=1)
    mixup_prob: float = Field(1.0, ge=0, le=1)
    enable_mixup: bool = True
    hsv_prob: float = Field(1.0, ge=0, le=1)
    flip_prob: float = Field(0.5, ge=0, le=1)
    degrees: float = Field(10.0, ge=0)
    translate: float = Field(0.1, ge=0)
    shear: float = Field(2.0, ge=0)
    mosaic_scale_min: float = Field(0.1, gt=0)
    mosaic_scale_max: float = Field(2.0, gt=0)
    mixup_scale_min: float = Field(0.5, gt=0)
    mixup_scale_max: float = Field(1.5, gt=0)

    # Evaluation and logging.
    eval_interval: int = Field(1, ge=1)
    print_interval: int = Field(20, ge=1)
    save_history_ckpt: bool = False
    test_conf: float = Field(0.01, ge=0, le=1)
    nmsthre: float = Field(0.65, gt=0, le=1)

    # Run.
    batch_size: int = Field(16, ge=1)
    devices: int = Field(1, ge=1)
    data_num_workers: int = Field(4, ge=0)
    fp16: bool = True
    occupy: bool = False
    cache: Literal["none", "ram", "disk"] = "none"
    baseline: str | None = Field(
        f"{BASELINE_WEIGHTS}:yolox_m.pth",
        description="What to start from: 'model:<name>' from the models folder, "
        "'weights:<file>' from the weights folder, or null for random initialisation.",
    )
    gpu_memory_cap_gb: float | None = Field(15.0, gt=0)
    export_onnx: bool = True
    onnx_opset: int = Field(11, ge=9)

    @model_validator(mode="after")
    def _check_ranges(self) -> TrainConfig:  # noqa: N804
        if self.mosaic_scale_min > self.mosaic_scale_max:
            raise ValueError("mosaic_scale_min must not exceed mosaic_scale_max.")
        if self.mixup_scale_min > self.mixup_scale_max:
            raise ValueError("mixup_scale_min must not exceed mixup_scale_max.")
        if self.baseline is not None and not _BASELINE.match(self.baseline):
            raise ValueError("baseline must be 'model:<name>', 'weights:<file>' or null.")
        return self

    def baseline_parts(self) -> tuple[str, str] | None:
        """The kind ('model' or 'weights') and the name of the baseline, or None."""
        if self.baseline is None:
            return None
        kind, _, name = self.baseline.partition(":")
        return kind, name
