# Detection models

Drop your ONNX detectors here. Every model that follows the convention below is registered
as an operator in the LightlyStudio GUI when `make start-pipeline` runs, under
**Plugins → "Detect: <name>"**.

## Convention

Two files per model, same stem:

```
models/
  yolox_m_slm_v4.onnx            the exported YOLOX model
  yolox_m_slm_v4_classes.txt     one class name per line, in model output order
```

Everything else in this folder is ignored (`.pth` checkpoints, notes, subfolders).

## What the operator does

- Runs on the images in the current view: the filtered grid, or your selection.
- Writes boxes into their own annotation source, `<name>_prediction` by default. The ground
  truth annotation source `annotation` is never written to.
- Creates the annotation classes from the classes file when they do not exist yet, reusing
  existing ones that differ only in case.
- Skips images that already have predictions in that annotation source unless `overwrite` is on.

## Model requirements

- Input `(1, 3, H, W)` float32, BGR, 0-255, YOLOX letterbox with pad value 114. H and W are
  read from the ONNX; a dynamic shape falls back to 640x640.
- Output `(1, N, 5 + num_classes)`. Both raw heads and heads exported with
  `decode_in_inference` are handled; the decoder is picked by inspecting the output.

Inference runs on CPU through `onnxruntime`. To use the GPU, install `onnxruntime-gpu`
into the workspace environment; the detector then prefers CUDA automatically, and
`PIPELINE_ONNX_DEVICE=cpu` forces CPU again.
