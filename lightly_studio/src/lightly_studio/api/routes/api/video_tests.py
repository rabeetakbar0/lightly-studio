"""API routes for running the TestApp gate-arm pipeline on uploaded videos."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile
from pydantic import BaseModel, Field

from lightly_studio.api.routes.api.status import HTTP_STATUS_CREATED
from lightly_studio.dataset import env
from lightly_studio.pipeline_training import model_catalog
from lightly_studio.video_tests import runner
from lightly_studio.video_tests.log_progress import VideoTestProgress
from lightly_studio.video_tests.runs import VideoTestRun

video_tests_router = APIRouter(tags=["video_tests"])

DEFAULT_LOG_LINES = 200
MAX_LOG_LINES = 5000


class VideoTestModelView(BaseModel):
    """A model a video test can run with."""

    name: str
    classes: list[str]


class VideoTestOptionsView(BaseModel):
    """Whether a video test can start, where its files go, and the models to pick from."""

    ready: bool
    problems: list[str]
    testapp_dir: str | None
    uploads_dir: str | None
    results_dir: str | None
    models: list[VideoTestModelView]


class VideoTestProgressView(BaseModel):
    """Where a running video test is."""

    total: int
    current_index: int
    current_video: str | None
    frames_done: int
    frames_total: int
    fps: float | None
    pass_number: int = Field(description="2 when the video is re-run with backup locations.")
    percent: float

    @classmethod
    def from_progress(cls, progress: VideoTestProgress) -> VideoTestProgressView:
        """Builds the view from the parsed log."""
        return cls(
            total=progress.total,
            current_index=progress.current_index,
            current_video=progress.current_video,
            frames_done=progress.frames_done,
            frames_total=progress.frames_total,
            fps=progress.fps,
            pass_number=progress.pass_number,
            percent=progress.percent,
        )


class VideoTestVideoView(BaseModel):
    """One uploaded video and its result."""

    file_name: str
    size_bytes: int
    status: str = Field(description="pending, processing, done or failed.")
    output_path: str | None
    summary: dict[str, str]
    benchmark: dict[str, str]


class VideoTestRunView(BaseModel):
    """One video test."""

    name: str
    model: str
    model_path: str
    input_dir: str
    output_dir: str
    status: str = Field(description="running, finished, failed or stopped.")
    created_at: str
    finished_at: str | None
    exit_code: int | None
    error: str | None
    videos: list[VideoTestVideoView]
    progress: VideoTestProgressView | None = Field(description="Set while the test runs.")


class VideoTestLogView(BaseModel):
    """The end of a video test's log."""

    text: str


@video_tests_router.get("/video_tests/options", response_model=VideoTestOptionsView)
def get_video_test_environment() -> VideoTestOptionsView:
    """The environment check, the upload and result folders and the models."""
    environment = runner.video_test_runner.environment()
    return VideoTestOptionsView(
        ready=environment.ready,
        problems=environment.problems,
        testapp_dir=environment.testapp_dir,
        uploads_dir=environment.uploads_dir,
        results_dir=environment.results_dir,
        models=[
            VideoTestModelView(name=model.name, classes=model.classes)
            for model in model_catalog.list_models(models_dir=env.LIGHTLY_STUDIO_MODELS_DIR)
        ],
    )


@video_tests_router.get("/video_tests", response_model=list[VideoTestRunView])
def list_video_tests() -> list[VideoTestRunView]:
    """All video tests, newest first."""
    return [_view(record) for record in runner.video_test_runner.list_runs()]


@video_tests_router.post(
    "/video_tests", response_model=VideoTestRunView, status_code=HTTP_STATUS_CREATED
)
def start_video_test(
    files: Annotated[list[UploadFile], File(description="The videos to process.")],
    model: Annotated[str, Form(description="A model of the models folder.")],
) -> VideoTestRunView:
    """Saves the videos to a new upload folder and runs the TestApp pipeline on them."""
    record = runner.video_test_runner.start(
        model=model, files=[(file.filename or "upload", file.file) for file in files]
    )
    return _view(record)


@video_tests_router.get("/video_tests/{name}", response_model=VideoTestRunView)
def get_video_test(name: str) -> VideoTestRunView:
    """One video test, with progress while it runs."""
    return _view(runner.video_test_runner.get_run(name=name))


@video_tests_router.get("/video_tests/{name}/log", response_model=VideoTestLogView)
def get_video_test_log(
    name: str, lines: int = Query(DEFAULT_LOG_LINES, ge=1, le=MAX_LOG_LINES)
) -> VideoTestLogView:
    """The last lines of the video test's log."""
    return VideoTestLogView(text=runner.video_test_runner.read_log(name=name, lines=lines))


@video_tests_router.post("/video_tests/{name}/stop", response_model=VideoTestRunView)
def stop_video_test(name: str) -> VideoTestRunView:
    """Terminates the pipeline."""
    return _view(runner.video_test_runner.stop(name=name))


def _view(record: VideoTestRun) -> VideoTestRunView:
    progress = None
    if record.is_active:
        progress = VideoTestProgressView.from_progress(
            runner.video_test_runner.progress(name=record.name)
        )
    return VideoTestRunView(
        name=record.name,
        model=record.model,
        model_path=record.model_path,
        input_dir=record.input_dir,
        output_dir=record.output_dir,
        status=record.status,
        created_at=record.created_at,
        finished_at=record.finished_at,
        exit_code=record.exit_code,
        error=record.error,
        videos=[
            VideoTestVideoView(
                file_name=video.file_name,
                size_bytes=video.size_bytes,
                status=video.status,
                output_path=video.output_path,
                summary=video.summary,
                benchmark=video.benchmark,
            )
            for video in record.videos
        ],
        progress=progress,
    )
