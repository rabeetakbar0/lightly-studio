---
name: fork-guide
description: Read before working in projects/ or on anything specific to this fork of LightlyStudio - the pipeline launcher, the models folder and Detect operators, GUI media upload and deletion, and how to run the backend and frontend on this machine. Covers the folder layout, start targets, upload folder, restart and rebuild rules, and the uv and Node PATH traps.
---

# Fork guide

This is a fork of LightlyStudio that grows into a full annotation and training pipeline.
Upstream code stays mergeable: fork-only code lives in `projects/` and in clearly separate
backend and frontend modules.

## Layout

- `projects/start_studio.py`: the launcher. Opens the persistent `pipeline` dataset, registers
  one `Detect: <name>` operator per model in `projects/models/`, starts the GUI on 127.0.0.1:8001.
- `projects/models/`: `<name>.onnx` plus `<name>_classes.txt` (one class per line, model output
  order). Weights are gitignored. `PIPELINE_MODELS_DIR` overrides the folder.
- `projects/pipeline/`: `model_registry` (folder scan), `yolox_onnx` (letterbox, decode, NMS
  through onnxruntime), `detect_operator` (IMAGE-scope operator writing to the annotation source
  `<name>_prediction`). Tests in `projects/tests`, lint config in `projects/ruff.toml`.
- `projects/uploads/` and `projects/lightly_studio_pipeline.db`: data added through the GUI.
  Both gitignored. `LIGHTLY_STUDIO_UPLOAD_DIR` is set by the launcher before the import.
- `projects/datasets/`: COCO datasets prepared from the GUI (Export → "Prepare COCO dataset"),
  one folder each: `train2017/`, `val2017/`, `annotations/instances_*.json`, `dataset.json`
  manifest, `README.md`. Images are hard links with bare file names; categories are 1-based.
  `LIGHTLY_STUDIO_DATASETS_DIR` points here (set by the launcher). Gitignored.
- Backend fork modules: `lightly_studio/src/lightly_studio/pipeline_datasets/` (`coco_dataset`:
  write, merge, list, delete folders; `service`: export from a collection, preview back into
  Studio as dataset `preview:<name>`) with routes `api/routes/api/pipeline_datasets.py`.
  The landing page skips `preview:` datasets. `lightly_studio/src/lightly_studio/upload/` (frame selection, frame
  extraction, upload and delete services) and routes `api/routes/api/media_upload.py`,
  `media_delete.py`. `resolvers/image_resolver/delete_many_with_dependents.py` is the ordered
  DuckDB cascade for image samples.
- Frontend fork modules: `lib/components/MediaDropZone`, `ConfirmDeleteDialog`,
  `DeleteAllImagesDialog`, `TrainPanel` (right side panel tab "Train": list, preview, merge,
  delete datasets; training itself is not wired yet), `ExportSamples/PrepareDatasetTab`; hooks
  `useMediaUpload`, `useDeleteImages`, `useDeleteAllImagesDialog`, `usePipelineDatasets`.

## Running

From `lightly_studio/`, with `uv` on PATH (`export PATH="$HOME/.local/bin:$PATH"`):

- `make start-pipeline`: keep existing data, append new drops.
- `make start-pipeline-clean`: delete the database and uploads first.
- `make start-example`: upstream example data, wipes its own database each run.

Always run fork scripts as `uv run python ../projects/<script>.py`. A bare script path makes uv
discover the workspace root, which has no `cpu` group, and it downloads the CUDA torch build.

Frontend dev server: `npm run dev` in `lightly_studio_view` reads `.env.development.local`
pointing at 127.0.0.1:8001. Node 24 from nvm must be first on PATH; the system Node 20 breaks
the Langium generator with "Object.groupBy is not a function".

## After a change

- Backend Python: restart the backend.
- Backend route or model: regenerate the client. Do not use `make export-schema`, it deletes
  the built bundle. Run `uv run src/lightly_studio/export_schema.py` from `lightly_studio/`
  with the bundle folder present, then `npm run generate-api-client` in `lightly_studio_view`.
- Frontend for the backend-served app: `make build-dirty-lightly_studio_view` from
  `lightly_studio/`, then restart the backend. The browser may cache the old bundle; hard-reload.
- `onnxruntime` is the `pipeline` dependency group, pinned to 1.19 on Python 3.9. GPU inference
  needs `onnxruntime-gpu`; the detector prefers CUDA when it is available.

## Rules

- Never write predictions into the ground truth annotation source `annotation`.
- Only files under the upload folder are deleted from disk; ingested paths elsewhere stay.
- User-facing wording follows the glossary skill: "annotation source", not "layer".
