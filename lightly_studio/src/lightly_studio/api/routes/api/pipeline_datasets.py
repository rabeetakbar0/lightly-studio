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
from lightly_studio.pipeline_datasets.val_split import SplitInfo
from lightly_studio.resolvers.image_filter import ImageFilter

pipeline_datasets_router = APIRouter(tags=["pipeline_datasets"])


class SplitCountsView(BaseModel):
    """Image and box counts of one split."""

    images: int
    boxes: int


class GroupCountsView(BaseModel):
    """How one similarity group was split."""

    train: int
    val: int
    test: int


class SplitInfoView(BaseModel):
    """How the hold-outs of a dataset were chosen."""

    strategy: str
    split_by: str
    val_fraction: float
    test_fraction: float
    groups: int = Field(description="Similarity groups used; 0 when nothing was clustered.")
    groups_requested: int | None
    group_counts: list[GroupCountsView]
    fallback_images: int = Field(description="Images split at random for lack of an embedding.")

    @classmethod
    def from_info(cls, info: SplitInfo) -> SplitInfoView:
        """Builds the view from the stored split info."""
        return cls(
            strategy=info.strategy,
            split_by=info.split_by,
            val_fraction=info.val_fraction,
            test_fraction=info.test_fraction,
            groups=info.groups,
            groups_requested=info.groups_requested,
            group_counts=[
                GroupCountsView(train=c.train, val=c.val, test=c.test) for c in info.group_counts
            ],
            fallback_images=info.fallback_images,
        )


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
    kind: str = Field(description="'train' or 'test' (a test-only dataset).")
    split: SplitInfoView | None

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
            kind=summary.kind,
            split=SplitInfoView.from_info(summary.split) if summary.split else None,
        )


class ExportPipelineDatasetRequest(BaseModel):
    """Which images and boxes to write, and how to split them."""

    name: str = Field(description="Folder name of the new dataset.")
    annotation_collection_id: UUID = Field(description="The annotation source to export.")
    image_filter: ImageFilter | None = Field(None, description="The current grid filter.")
    val_fraction: float = Field(0.0, ge=0, lt=1)
    test_fraction: float = Field(0.0, ge=0, lt=1)
    split_by: Literal["tag", "image"] = "tag"
    split_strategy: Literal["random", "embedding"] = Field(
        "embedding",
        description="'embedding' spreads the hold-outs over similarity groups, 'random' does not.",
    )
    groups: int | None = Field(
        None, ge=2, le=4096, description="Similarity groups; null picks the count automatically."
    )


class MergePipelineDatasetsRequest(BaseModel):
    """Which datasets to merge into a new one."""

    name: str
    sources: list[str] = Field(min_length=2)


class BuildTestDatasetRequest(BaseModel):
    """Which datasets to draw a test dataset from, and how much."""

    name: str
    sources: list[str] = Field(min_length=1)
    fraction: float | None = Field(None, gt=0, lt=1, description="Share of the pooled images.")
    count: int | None = Field(None, ge=1, description="Number of images instead of a share.")
    split_strategy: Literal["random", "embedding"] = "embedding"
    groups: int | None = Field(None, ge=2, le=4096)


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
    if body.val_fraction + body.test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must sum to below 1.")
    summary = service.export_dataset(
        session=session,
        collection_id=collection.collection_id,
        name=body.name.strip(),
        annotation_collection_id=body.annotation_collection_id,
        image_filter=body.image_filter,
        val_fraction=body.val_fraction,
        test_fraction=body.test_fraction,
        split_by=body.split_by,
        split_strategy=body.split_strategy,
        groups=body.groups,
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
    "/pipeline_datasets/build_test",
    response_model=PipelineDatasetView,
    status_code=HTTP_STATUS_CREATED,
)
def build_test_dataset(session: SessionDep, body: BuildTestDatasetRequest) -> PipelineDatasetView:
    """Draws a test-only dataset from all images of the given datasets."""
    summary = service.build_test_dataset(
        session=session,
        name=body.name.strip(),
        sources=body.sources,
        fraction=body.fraction,
        count=body.count,
        split_strategy=body.split_strategy,
        groups=body.groups,
    )
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
