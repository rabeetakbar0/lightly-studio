"""YOLOX inference through onnxruntime.

The pre- and post-processing follow YOLOX's `ValTransform` and `yolo_head.decode_outputs`,
so a model exported with `tools/export_onnx.py` works with or without `decode_in_inference`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

PAD_VALUE = 114  # YOLOX letterbox fill value.
DEFAULT_INPUT_SIZE = (640, 640)  # (h, w), used when the ONNX input shape is dynamic.
STRIDES = (8, 16, 32)
NMS_IOU = 0.65  # YOLOX exp.nmsthre.
_NCHW_RANK = 4
# A decoded head emits pixel sizes; any real box is wider than this many pixels.
_MIN_DECODED_SIZE = 8


@dataclass(frozen=True)
class Detection:
    """One box in pixel coordinates of the original image.

    Attributes:
        class_index: Index into the model's class list.
        score: Objectness times class probability, in [0, 1].
        x1, y1, x2, y2: Corners, clipped to the image.
    """

    class_index: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


class YoloxDetector:
    """Runs one YOLOX ONNX model.

    Attributes:
        input_size: The (h, w) the model expects.
    """

    def __init__(self, onnx_path: Path, providers: list[str] | None = None) -> None:
        """Loads the model.

        Args:
            onnx_path: The model file.
            providers: onnxruntime execution providers. Defaults to CUDA when available,
                else CPU.
        """
        self._session = ort.InferenceSession(
            str(onnx_path), providers=providers or default_providers()
        )
        self._input_name = self._session.get_inputs()[0].name
        self.input_size = input_size_of(session=self._session)

    def predict(
        self, image: NDArray[Any], score_threshold: float, nms_iou: float = NMS_IOU
    ) -> list[Detection]:
        """Detects objects in one BGR image of shape (H, W, 3).

        Args:
            image: The image, uint8 BGR as OpenCV reads it.
            score_threshold: Detections below this score are dropped.
            nms_iou: IoU above which two boxes of the same class are duplicates.

        Returns:
            The kept detections, in no particular order.
        """
        tensor, ratio = letterbox(image=image, input_size=self.input_size)
        raw = self._session.run(None, {self._input_name: tensor[None]})[0][0]
        if not outputs_are_decoded(raw=raw):
            raw = decode_outputs(raw=raw, input_size=self.input_size)
        height, width = image.shape[:2]
        return postprocess(
            raw=raw,
            ratio=ratio,
            image_size=(height, width),
            score_threshold=score_threshold,
            nms_iou=nms_iou,
        )


def default_providers() -> list[str]:
    """CUDA first when onnxruntime can use it, else CPU. PIPELINE_ONNX_DEVICE=cpu forces CPU."""
    if os.environ.get("PIPELINE_ONNX_DEVICE", "").lower() == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return ["CPUExecutionProvider"]
    if hasattr(ort, "preload_dlls"):
        # Loads CUDA and cuDNN from the nvidia-* pip packages when they are installed.
        ort.preload_dlls(cuda=True, cudnn=True)
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def input_size_of(session: ort.InferenceSession) -> tuple[int, int]:
    """(h, w) the model expects, read from its input shape."""
    shape = session.get_inputs()[0].shape
    if len(shape) == _NCHW_RANK and all(isinstance(value, int) for value in shape[2:]):
        return int(shape[2]), int(shape[3])
    return DEFAULT_INPUT_SIZE


def letterbox(image: NDArray[Any], input_size: tuple[int, int]) -> tuple[NDArray[Any], float]:
    """Resizes keeping the aspect ratio and pads bottom and right.

    Returns:
        The CHW float32 tensor and the resize ratio.
    """
    ratio = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    new_h, new_w = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
    padded = np.full((input_size[0], input_size[1], 3), PAD_VALUE, dtype=np.uint8)
    padded[:new_h, :new_w] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


def outputs_are_decoded(raw: NDArray[Any]) -> bool:
    """True when the model already emits pixel boxes.

    A raw head emits log-space sizes, which can be negative and are small; pixel sizes are
    never negative and exceed a few pixels for any real box.
    """
    return bool(raw[:, 2:4].min() >= 0 and raw[:, 2:4].max() > _MIN_DECODED_SIZE)


def decode_outputs(raw: NDArray[Any], input_size: tuple[int, int]) -> NDArray[Any]:
    """Turns raw head outputs (N, 5 + C) into center-size pixel boxes, like YOLOX's decoder."""
    grids = []
    strides = []
    for stride in STRIDES:
        hs, ws = input_size[0] // stride, input_size[1] // stride
        yv, xv = np.meshgrid(np.arange(hs), np.arange(ws), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((grid.shape[0], 1), stride))
    grid_xy = np.concatenate(grids).astype(np.float32)
    grid_stride = np.concatenate(strides).astype(np.float32)

    out = raw.copy()
    out[:, :2] = (raw[:, :2] + grid_xy) * grid_stride
    out[:, 2:4] = np.exp(raw[:, 2:4]) * grid_stride
    return out


def postprocess(
    raw: NDArray[Any],
    ratio: float,
    image_size: tuple[int, int],
    score_threshold: float,
    nms_iou: float,
) -> list[Detection]:
    """Scores, thresholds, scales back to the image and applies per-class NMS."""
    cx, cy, w, h = (raw[:, i] / ratio for i in range(4))
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    height, width = image_size
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)

    class_scores = raw[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    scores = raw[:, 4] * class_scores[np.arange(raw.shape[0]), class_ids]

    detections: list[Detection] = []
    for class_id in np.unique(class_ids):
        mask = (class_ids == class_id) & (scores >= score_threshold)
        if not mask.any():
            continue
        class_boxes, class_box_scores = boxes[mask], scores[mask]
        for i in nms(boxes=class_boxes, scores=class_box_scores, iou_threshold=nms_iou):
            x1, y1, x2, y2 = class_boxes[i]
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                detections.append(
                    Detection(
                        class_index=int(class_id),
                        score=float(class_box_scores[i]),
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    )
                )
    return detections


def nms(boxes: NDArray[Any], scores: NDArray[Any], iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression on xyxy boxes; returns the indices to keep."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        inter = np.maximum(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0) * (
            np.maximum(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0)
        )
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep
