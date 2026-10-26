"""API routes for adding images and videos to a collection by file upload."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi import Path as FastAPIPath
from pydantic import BaseModel, Field

from lightly_studio.api.routes.api.collection import get_and_validate_collection_id
from lightly_studio.api.routes.api.status import HTTP_STATUS_CREATED
from lightly_studio.database.db_manager import SessionDep
from lightly_studio.models.collection import CollectionTable
from lightly_studio.upload import frame_selection, media_upload
from lightly_studio.upload.frame_selection import FrameSelectionConfig

media_upload_router = APIRouter(tags=["media_upload"])


class VideoUploadView(BaseModel):
    """Outcome of one uploaded video."""

    file_name: str
    duration_seconds: float
    frames_extracted: int
    frames_added: int
    tag: str


class MediaUploadView(BaseModel):
    """Outcome of an upload request."""

    added_sample_ids: list[UUID]
    skipped: int
    embedded: bool
    warnings: list[str]
    videos: list[VideoUploadView]


class FramePlanRequest(BaseModel):
    """Asks how many frames the selection settings yield for videos of given lengths."""

    durations_seconds: list[float] = Field(description="One duration per video.")
    edge_window_seconds: float = Field(20.0, ge=0)
    edge_interval_seconds: float = Field(1.0, gt=0)
    middle_frames: int = Field(35, ge=0)
    max_frames: int = Field(100, ge=1)


class FramePlanView(BaseModel):
    """Frame counts for a FramePlanRequest."""

    frame_counts: list[int] = Field(description="One count per requested duration.")
    total_frames: int


@media_upload_router.post(
    "/collections/{collection_id}/uploads/images",
    response_model=MediaUploadView,
    status_code=HTTP_STATUS_CREATED,
)
def upload_images(
    session: SessionDep,
    collection: Annotated[
        CollectionTable,
        FastAPIPath(title="collection Id"),
        Depends(get_and_validate_collection_id),
    ],
    files: Annotated[list[UploadFile], File(description="Image files to add.")],
    embed: Annotated[bool, Form(description="Compute embeddings for the new samples.")] = True,
    tag: Annotated[str, Form(description="Tag for the new samples; empty for none.")] = "",
) -> MediaUploadView:
    """Adds uploaded image files to the collection as image samples."""
    outcome = media_upload.add_uploaded_images(
        session=session,
        collection_id=collection.collection_id,
        files=[(file.filename or "upload", file.file) for file in files],
        embed=embed,
        tag=tag.strip(),
    )
    return MediaUploadView.model_validate(outcome, from_attributes=True)


@media_upload_router.post(
    "/collections/{collection_id}/uploads/videos",
    response_model=MediaUploadView,
    status_code=HTTP_STATUS_CREATED,
)
def upload_videos(  # noqa: PLR0913
    session: SessionDep,
    collection: Annotated[
        CollectionTable,
        FastAPIPath(title="collection Id"),
        Depends(get_and_validate_collection_id),
    ],
    files: Annotated[list[UploadFile], File(description="Video files to extract frames from.")],
    edge_window_seconds: Annotated[float, Form(ge=0)] = 20.0,
    edge_interval_seconds: Annotated[float, Form(gt=0)] = 1.0,
    middle_frames: Annotated[int, Form(ge=0)] = 35,
    max_frames: Annotated[int, Form(ge=1)] = 100,
    embed: Annotated[bool, Form(description="Compute embeddings for the new samples.")] = True,
    tags: Annotated[
        list[str] | None,
        Form(description="One tag per video, in file order. Defaults to the video name."),
    ] = None,
) -> MediaUploadView:
    """Extracts frames from uploaded videos and adds them to the collection as images.

    The start and the end of each video are sampled every `edge_interval_seconds` for
    `edge_window_seconds`, the middle gets `middle_frames` evenly spaced frames, and the
    total is capped at `max_frames`.
    """
    config = FrameSelectionConfig(
        edge_window_seconds=edge_window_seconds,
        edge_interval_seconds=edge_interval_seconds,
        middle_frames=middle_frames,
        max_frames=max_frames,
    )
    outcome = media_upload.add_uploaded_videos(
        session=session,
        collection_id=collection.collection_id,
        files=[(file.filename or "upload", file.file) for file in files],
        config=config,
        embed=embed,
        tags=[tag.strip() for tag in tags] if tags is not None else None,
    )
    return MediaUploadView.model_validate(outcome, from_attributes=True)


@media_upload_router.post("/uploads/frame_plan", response_model=FramePlanView)
def plan_video_frames(body: FramePlanRequest) -> FramePlanView:
    """Returns how many frames the given settings extract from videos of the given lengths."""
    config = FrameSelectionConfig(
        edge_window_seconds=body.edge_window_seconds,
        edge_interval_seconds=body.edge_interval_seconds,
        middle_frames=body.middle_frames,
        max_frames=body.max_frames,
    )
    counts = [
        len(frame_selection.select_timestamps(duration_seconds=duration, config=config))
        for duration in body.durations_seconds
    ]
    return FramePlanView(frame_counts=counts, total_frames=sum(counts))
