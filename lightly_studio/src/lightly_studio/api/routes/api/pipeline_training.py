"""API routes for training runs on the prepared datasets."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from lightly_studio.api.routes.api.status import HTTP_STATUS_CREATED
from lightly_studio.dataset import env
from lightly_studio.pipeline_training import model_catalog, runner, runs
from lightly_studio.pipeline_training.config import BASELINE_MODEL, BASELINE_WEIGHTS, TrainConfig
from lightly_studio.pipeline_training.log_metrics import TrainingProgress
from lightly_studio.pipeline_training.model_catalog import MODEL_PRESETS
from lightly_studio.pipeline_training.runner import TrainingEnvironment
from lightly_studio.pipeline_training.runs import RunRecord

pipeline_training_router = APIRouter(tags=["pipeline_training"])

DEFAULT_LOG_LINES = 200
MAX_LOG_LINES = 5000


class TrainingEnvironmentView(BaseModel):
    """Whether training can start, and with what."""

    yolox_dir: str | None
    python: str | None
    weights: list[str]
    problems: list[str]
    ready: bool

    @classmethod
    def from_environment(cls, environment: TrainingEnvironment) -> TrainingEnvironmentView:
        """Builds the view from the runner's check."""
        return cls(
            yolox_dir=environment.yolox_dir,
            python=environment.python,
            weights=environment.weights,
            problems=environment.problems,
            ready=environment.ready,
        )


class BaselineView(BaseModel):
    """Something a run can start from."""

    value: str = Field(description="The TrainConfig.baseline value.")
    label: str
    group: str = Field(description="'Trained models' or 'Stock YOLOX'.")
    depth: float | None
    width: float | None
    classes: list[str]
    available: bool = Field(description="False for a model without a checkpoint.")


class TrainingOptionsView(BaseModel):
    """What the train dialog needs: defaults, model presets, baselines and the environment."""

    defaults: TrainConfig
    presets: dict[str, list[float]] = Field(description="Depth and width per YOLOX size.")
    baselines: list[BaselineView]
    environment: TrainingEnvironmentView


class StartTrainingRequest(BaseModel):
    """A new run."""

    name: str = Field(description="Run and model name.")
    dataset: str = Field(description="A prepared dataset.")
    config: TrainConfig


class EpochEvaluationView(BaseModel):
    """COCO scores after one epoch, as fractions."""

    epoch: int
    ap50: float | None
    ap50_95: float | None


class TrainingProgressView(BaseModel):
    """Where the run is, read from its log."""

    epoch: int
    max_epoch: int
    iteration: int
    max_iteration: int
    eta: str
    best_ap50_95: float | None
    evaluations: list[EpochEvaluationView]
    last_line: str

    @classmethod
    def from_progress(cls, progress: TrainingProgress) -> TrainingProgressView:
        """Builds the view from the parsed log."""
        return cls(
            epoch=progress.epoch,
            max_epoch=progress.max_epoch,
            iteration=progress.iteration,
            max_iteration=progress.max_iteration,
            eta=progress.eta,
            best_ap50_95=progress.best_ap50_95,
            evaluations=[
                EpochEvaluationView(epoch=e.epoch, ap50=e.ap50, ap50_95=e.ap50_95)
                for e in progress.evaluations
            ],
            last_line=progress.last_line,
        )


class TrainingRunView(BaseModel):
    """One run with its progress."""

    name: str
    dataset: str
    classes: list[str]
    config: TrainConfig
    status: str
    created_at: str
    finished_at: str | None
    exit_code: int | None
    onnx_path: str | None
    error: str | None
    progress: TrainingProgressView

    @classmethod
    def from_record(cls, record: RunRecord, progress: TrainingProgress) -> TrainingRunView:
        """Builds the view from the manifest and the parsed log."""
        return cls(
            name=record.name,
            dataset=record.dataset,
            classes=record.classes,
            config=TrainConfig.model_validate(record.config),
            status=record.status,
            created_at=record.created_at,
            finished_at=record.finished_at,
            exit_code=record.exit_code,
            onnx_path=record.onnx_path,
            error=record.error,
            progress=TrainingProgressView.from_progress(progress),
        )


class TrainingLogView(BaseModel):
    """The tail of a run's log."""

    text: str


@pipeline_training_router.get("/pipeline_training/options", response_model=TrainingOptionsView)
def get_training_options() -> TrainingOptionsView:
    """The defaults of the v4 recipe, the model size presets and the environment check."""
    environment = runner.training_runner.environment()
    return TrainingOptionsView(
        defaults=TrainConfig(),
        presets={name: list(values) for name, values in MODEL_PRESETS.items()},
        baselines=_baselines(weights=environment.weights),
        environment=TrainingEnvironmentView.from_environment(environment),
    )


@pipeline_training_router.get("/pipeline_training/runs", response_model=list[TrainingRunView])
def list_training_runs() -> list[TrainingRunView]:
    """All runs, newest first."""
    return [_view(record) for record in runner.training_runner.list_runs()]


@pipeline_training_router.post(
    "/pipeline_training/runs", response_model=TrainingRunView, status_code=HTTP_STATUS_CREATED
)
def start_training_run(body: StartTrainingRequest) -> TrainingRunView:
    """Writes the exp file and starts tools/train.py in a new run folder."""
    record = runner.training_runner.start(
        name=body.name.strip(), dataset=body.dataset, config=body.config
    )
    return _view(record)


@pipeline_training_router.get("/pipeline_training/runs/{name}", response_model=TrainingRunView)
def get_training_run(name: str) -> TrainingRunView:
    """One run with the progress read from its log."""
    return _view(runner.training_runner.get_run(name=name))


@pipeline_training_router.get("/pipeline_training/runs/{name}/log", response_model=TrainingLogView)
def get_training_log(
    name: str, lines: int = Query(DEFAULT_LOG_LINES, ge=1, le=MAX_LOG_LINES)
) -> TrainingLogView:
    """The last lines of the run's log."""
    record = runner.training_runner.get_run(name=name)
    run_dir = runs.run_dir(runs_dir=env.LIGHTLY_STUDIO_RUNS_DIR, name=record.name)
    return TrainingLogView(text=runs.read_log_tail(run_dir=run_dir, lines=lines))


@pipeline_training_router.post(
    "/pipeline_training/runs/{name}/stop", response_model=TrainingRunView
)
def stop_training_run(name: str) -> TrainingRunView:
    """Terminates the training process."""
    return _view(runner.training_runner.stop(name=name))


@pipeline_training_router.delete("/pipeline_training/runs/{name}")
def delete_training_run(name: str) -> dict[str, str]:
    """Removes the run folder; an exported model stays in the models folder."""
    runner.training_runner.delete(name=name)
    return {"status": "deleted"}


def _baselines(weights: list[str]) -> list[BaselineView]:
    baselines = [
        BaselineView(
            value=f"{BASELINE_MODEL}:{model.name}",
            label=model.name,
            group="Trained models",
            depth=model.depth,
            width=model.width,
            classes=model.classes,
            available=model.checkpoint_path is not None,
        )
        for model in model_catalog.list_models(models_dir=env.LIGHTLY_STUDIO_MODELS_DIR)
    ]
    for file_name in weights:
        size = model_catalog.size_from_name(name=file_name)
        baselines.append(
            BaselineView(
                value=f"{BASELINE_WEIGHTS}:{file_name}",
                label=file_name,
                group="Stock YOLOX",
                depth=size[0] if size else None,
                width=size[1] if size else None,
                classes=[],
                available=True,
            )
        )
    return baselines


def _view(record: RunRecord) -> TrainingRunView:
    run_dir = runs.run_dir(runs_dir=env.LIGHTLY_STUDIO_RUNS_DIR, name=record.name)
    return TrainingRunView.from_record(record=record, progress=runs.read_progress(run_dir=run_dir))
