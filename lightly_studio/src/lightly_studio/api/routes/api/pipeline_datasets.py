"""API routes for the prepared training datasets in the datasets folder."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi import Path as FastAPIPath
from pydantic import BaseModel, Field

from lightly_studio.api.routes.api.collection import get_and_validate_collection_id
from lightly_studio.api.routes.api.status import HTTP_STATUS_CREATED
from lightly_studio.database.db_manager import SessionDep
from lightly_studio.models.collection import CollectionTable
from lightly_studio.pipeline_datasets import service
from lightly_studio.pipeline_datasets.coco_dataset import DatasetSummary
from lightly_studio.resolvers.image_filter import ImageFilter

pipeline_datasets_router = APIRouter(tags=["pipeline_datasets"])


class SplitCountsView(BaseModel):
    """Image and box counts of one split."""

    images: int
    boxes: int


class PipelineDatasetView(BaseModel):
    """One prepared dataset folder."""

    name: str
    path: str
    created_at: str
    classes: list[str]
    splits: dict[str, SplitCountsView]
    sources: list[str]
    annotation_source: str
    size_bytes: int
    images: int
    boxes: int

    @classmethod
    def from_summary(cls, summary: DatasetSummary) -> PipelineDatasetView:
        """Builds the view from the service's summary."""
        return cls(
            name=summary.name,
            path=summary.path,
            created_at=summary.created_at,
            classes=summary.classes,
            splits={
                split: SplitCountsView(images=counts.images, boxes=counts.boxes)
                for split, counts in summary.splits.items()
            },
            sources=summary.sources,
            annotation_source=summary.annotation_source,
            size_bytes=summary.size_bytes,
            images=summary.images,
            boxes=summary.boxes,
        )


class ExportPipelineDatasetRequest(BaseModel):
    """Which images and boxes to write, and how to split them."""

    name: str = Field(description="Folder name of the new dataset.")
    annotation_collection_id: UUID = Field(description="The annotation source to export.")
    image_filter: ImageFilter | None = Field(None, description="The current grid filter.")
    val_fraction: float = Field(0.0, ge=0, lt=1)
    split_by: Literal["tag", "image"] = "tag"


class MergePipelineDatasetsRequest(BaseModel):
    """Which datasets to merge into a new one."""

    name: str
    sources: list[str] = Field(min_length=2)


class PreviewPipelineDatasetView(BaseModel):
    """The Studio collection that shows the dataset."""

    collection_id: UUID


@pipeline_datasets_router.get("/pipeline_datasets", response_model=list[PipelineDatasetView])
def list_pipeline_datasets() -> list[PipelineDatasetView]:
    """Lists the prepared datasets, newest first."""
    return [PipelineDatasetView.from_summary(summary) for summary in service.list_datasets()]


@pipeline_datasets_router.post(
    "/collections/{collection_id}/pipeline_datasets/export",
    response_model=PipelineDatasetView,
    status_code=HTTP_STATUS_CREATED,
)
def export_pipeline_dataset(
    session: SessionDep,
    collection: Annotated[
        CollectionTable,
        FastAPIPath(title="collection Id"),
        Depends(get_and_validate_collection_id),
    ],
    body: ExportPipelineDatasetRequest,
) -> PipelineDatasetView:
    """Writes the images in the current view and their boxes as a COCO dataset folder."""
    summary = service.export_dataset(
        session=session,
        collection_id=collection.collection_id,
        name=body.name.strip(),
        annotation_collection_id=body.annotation_collection_id,
        image_filter=body.image_filter,
        val_fraction=body.val_fraction,
        split_by=body.split_by,
    )
    return PipelineDatasetView.from_summary(summary)


@pipeline_datasets_router.post(
    "/pipeline_datasets/merge",
    response_model=PipelineDatasetView,
    status_code=HTTP_STATUS_CREATED,
)
def merge_pipeline_datasets(body: MergePipelineDatasetsRequest) -> PipelineDatasetView:
    """Merges datasets into a new one; each source keeps its own split."""
    summary = service.merge_datasets(name=body.name.strip(), sources=body.sources)
    return PipelineDatasetView.from_summary(summary)


@pipeline_datasets_router.post(
    "/pipeline_datasets/{name}/preview", response_model=PreviewPipelineDatasetView
)
def preview_pipeline_dataset(session: SessionDep, name: str) -> PreviewPipelineDatasetView:
    """Loads the dataset into Studio as "preview:<name>" and returns that collection."""
    return PreviewPipelineDatasetView(
        collection_id=service.preview_dataset(session=session, name=name)
    )


@pipeline_datasets_router.delete("/pipeline_datasets/{name}")
def delete_pipeline_dataset(name: str) -> dict[str, str]:
    """Removes the dataset folder."""
    service.delete_dataset(name=name)
    return {"status": "deleted"}
