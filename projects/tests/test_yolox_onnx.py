import numpy as np
import pytest

from pipeline import yolox_onnx


def test_letterbox() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    tensor, ratio = yolox_onnx.letterbox(image=image, input_size=(64, 64))
    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == np.float32
    assert ratio == pytest.approx(0.32)
    # Image occupies the top 32 rows, the rest is the pad value.
    assert tensor[0, 0, 0] == 255
    assert tensor[0, 40, 0] == yolox_onnx.PAD_VALUE


def test_outputs_are_decoded() -> None:
    raw_head = np.array([[0.1, 0.2, -1.5, 0.3, 0.9, 0.8]], dtype=np.float32)
    pixel_boxes = np.array([[100.0, 120.0, 40.0, 30.0, 0.9, 0.8]], dtype=np.float32)
    assert yolox_onnx.outputs_are_decoded(raw=raw_head) is False
    assert yolox_onnx.outputs_are_decoded(raw=pixel_boxes) is True


def test_decode_outputs() -> None:
    # 16x16 input: grid cells 4 + 1 + 0 = 5 anchors (stride 8: 2x2, stride 16: 1x1, stride 32: 0).
    raw = np.zeros((5, 6), dtype=np.float32)
    raw[1, :4] = [0.5, 0.0, np.log(2.0), np.log(3.0)]  # Cell (x=1, y=0) at stride 8.
    out = yolox_onnx.decode_outputs(raw=raw, input_size=(16, 16))
    assert out.shape == (5, 6)
    assert out[1, :4] == pytest.approx([12.0, 0.0, 16.0, 24.0])
    assert out[0, :4] == pytest.approx([0.0, 0.0, 8.0, 8.0])


def test_postprocess() -> None:
    # Two overlapping boxes of class 0 and one box of class 1, at ratio 0.5 (image is 2x bigger).
    raw = np.array(
        [
            [50, 50, 20, 20, 0.9, 0.9, 0.1],
            [52, 52, 20, 20, 0.8, 0.9, 0.1],
            [100, 100, 10, 10, 0.9, 0.2, 0.9],
            [10, 10, 10, 10, 0.1, 0.9, 0.1],  # Below threshold.
        ],
        dtype=np.float32,
    )
    detections = yolox_onnx.postprocess(
        raw=raw, ratio=0.5, image_size=(400, 400), score_threshold=0.5, nms_iou=0.65
    )
    by_class = {d.class_index: d for d in detections}
    assert len(detections) == 2
    assert by_class[0].score == pytest.approx(0.81)
    assert (by_class[0].x1, by_class[0].y1, by_class[0].x2, by_class[0].y2) == (80, 80, 120, 120)
    assert (by_class[1].x1, by_class[1].y1) == (190, 190)


def test_nms() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
    assert yolox_onnx.nms(boxes=boxes, scores=scores, iou_threshold=0.5) == [1, 2]
