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
  order). Optional `<name>.pth` (YOLOX checkpoint; makes the model a possible baseline) and
  `<name>.json` (model card: dataset, baseline, config, scores; written by the pipeline). Weights
  are gitignored. `PIPELINE_MODELS_DIR` overrides the folder; `LIGHTLY_STUDIO_MODELS_DIR` is set
  from it by the launcher. v4 and v3_1 checkpoints are symlinked from SLMmodelv2/YOLOX_outputs.
- `projects/pipeline/`: `model_registry` (folder scan), `yolox_onnx` (letterbox, decode, NMS
  through onnxruntime), `detect_operator` (IMAGE-scope operator writing to the annotation source
  `<name>_prediction`). Tests in `projects/tests`, lint config in `projects/ruff.toml`.
- `projects/uploads/` and `projects/lightly_studio_pipeline.db`: data added through the GUI.
  Both gitignored. `LIGHTLY_STUDIO_UPLOAD_DIR` is set by the launcher before the import.
- `projects/datasets/`: COCO datasets prepared from the GUI (Export → "Prepare COCO dataset"),
  one folder each: `train2017/`, `val2017/`, `test2017/`, `annotations/instances_*.json`,
  `dataset.json` manifest (`kind` train|test, `split` info: strategy, fractions, groups, per
  group counts, fallback count), `README.md`. Images are hard links with bare file names;
  categories are 1-based; each COCO image entry carries the Studio `sample_id`. Hold-outs are
  spread over k-means groups of the image embeddings (`pipeline_datasets/val_split.py`:
  `build_units`, `holdout_split`, `choose_groups` = sqrt(n/2) bounded by hold-out count and
  512, `allocate` largest-remainder); `service.embedding_vectors` computes missing embeddings
  first. "Build test dataset" (`load_pool` + `write_test_dataset`) draws a test-only dataset
  from existing ones.
  `LIGHTLY_STUDIO_DATASETS_DIR` points here (set by the launcher). Gitignored.
- `projects/runs/<name>/`: one folder per training run started from the GUI: `run.json`
  (config, status, classes, command), `exp.py` (the generated YOLOX exp file), `log.txt`
  (stdout of `tools/train.py`, then of the ONNX export), `train/` (YOLOX checkpoints and its
  own log), `export/` (the ONNX file and class list, also copied to the models folder).
  `LIGHTLY_STUDIO_RUNS_DIR` points here. Gitignored.
- `projects/weights/`: pretrained `.pth` checkpoints a run can start from
  (`LIGHTLY_STUDIO_WEIGHTS_DIR`). `yolox_m.pth` is a symlink into SLMmodelv2. Gitignored.
- Training needs `LIGHTLY_STUDIO_YOLOX_DIR` (the YOLOX checkout) and
  `LIGHTLY_STUDIO_TRAIN_PYTHON` (the interpreter with YOLOX, torch and CUDA; the SLMmodelv2
  venv, Python 3.12) in `lightly_studio/.env`. The backend only launches subprocesses; it never
  imports torch for training. One run at a time. The training process is a child of the
  backend: stopping the backend kills it, and such runs show as failed after a restart.
- Video tests need `LIGHTLY_STUDIO_TESTAPP_DIR` (the TestApp checkout with
  `App/test_pipeline.py`) and `LIGHTLY_STUDIO_TESTAPP_PYTHON` (its `.venv/bin/python`,
  onnxruntime-gpu) in `lightly_studio/.env`; `LIGHTLY_STUDIO_TESTAPP_UPLOADS_DIR` defaults to
  `<TESTAPP_DIR>/uploads`. A run saves the videos to `uploads/<YYYYmmdd-HHMMSS>/` and starts
  `python -u App/test_pipeline.py` with `cwd` = TestApp and `GATEARM_MODEL_PATH`,
  `GATEARM_INPUT_DIR`, `GATEARM_OUTPUT_DIR` set (the TestApp scripts read these, their old
  hard-coded paths are the defaults). `uploads/results/<name>/` holds the processed videos,
  `log.txt` and `run.json`. Progress is parsed from the lines the scripts already log
  (`[i/N] name`, `Progress f/F`, the backup-locations re-run, `Detection summary:`,
  `Done. Annotated video:`, `Failed to process`), so keep those lines when editing the TestApp.
  One video test at a time, independent of training. The pipeline also updates
  `TestApp/Logs/gatearm_locations.json`, as a manual run does.
- Backend fork modules: `lightly_studio/src/lightly_studio/pipeline_training/` (`config`:
  `TrainConfig` with the v4 recipe as defaults, `baseline` = `model:<name>` | `weights:<file>` |
  null; `model_catalog`: models folder scan, checkpoints, model cards, size presets; `exp_writer`;
  `log_metrics`: epoch/ETA/AP parser; `runs`: run folders; `runner`: subprocess start, stop,
  export, reconcile; `export_onnx_script.py` runs under the training interpreter, excluded from
  mypy) with routes `api/routes/api/pipeline_training.py` (`/pipeline_training/options` incl.
  baselines, `/pipeline_training/runs...`) and `pipeline_models.py` (`GET /pipeline_models`). `lightly_studio/src/lightly_studio/pipeline_datasets/` (`coco_dataset`:
  write, merge, list, delete folders; `service`: export from a collection, preview back into
  Studio as dataset `preview:<name>`) with routes `api/routes/api/pipeline_datasets.py`.
  The landing page skips `preview:` datasets. `lightly_studio/src/lightly_studio/upload/` (frame selection, frame
  extraction, upload and delete services) and routes `api/routes/api/media_upload.py`,
  `media_delete.py`. `lightly_studio/src/lightly_studio/video_tests/` (`log_progress`: log
  parser; `runs`: `run.json` records; `runner`: upload, launch, supervise, stop, reconcile) with
  routes `api/routes/api/video_tests.py` (`/video_tests/options`, `/video_tests`,
  `/video_tests/{name}`, `/log`, `/stop`). `resolvers/image_resolver/delete_many_with_dependents.py` is the ordered
  DuckDB cascade for image samples.
  `upload/annotation_source_delete.py` removes an annotation source (annotations via
  `annotation_resolver.delete_annotations_by_ids`, evaluation runs incl. class metrics, its tags,
  coverage, embedding-model and export-job rows, then the collection) and unused annotation
  classes; route `POST /collections/{id}/annotation_sources/delete`; `delete_all` calls it for
  every source. An annotation's source is its sample's `collection_id`, not a column on
  `annotation_base`.
- Frontend fork modules: `lib/components/MediaDropZone`, `ConfirmDeleteDialog`,
  `DeleteAllImagesDialog`, `TrainPanel` (right side panel tab "Train": datasets list, preview, merge,
  delete; `TrainModelDialog` with one tab per parameter group driven by `trainFields.ts`;
  `RunsSection`, `RunRow`, `RunLogDialog` polling active runs; `RunsSection` offers
  `TestModelDialog` when a training finishes; `BuildTestDatasetDialog`), `ModelsPanel` (side panel tab
  "Models": `ModelRow`, `CompareModelsDialog` over `compareRows.ts`, "Train from" opens
  `TrainModelDialog` with `initialConfig`), `VideoTestPanel` (side panel tab "Video": drop
  area, model select, `VideoTestRunRow` with progress bar and output paths, `VideoTestLogDialog`;
  drops on the panel stop propagation so the grid's window-level drop zone ignores them),
  `ExportSamples/PrepareDatasetTab`; hooks `useVideoTests`,
  `useMediaUpload`, `useDeleteImages`, `useDeleteAllImagesDialog`, `usePipelineDatasets`,
  `usePipelineTraining`, `usePipelineModels`.

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
