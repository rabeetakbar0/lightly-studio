"""API routes for removing images from a collection."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi import Path as FastAPIPath
from pydantic import BaseModel, Field

from lightly_studio.api.routes.api.collection import get_and_validate_collection_id
from lightly_studio.database.db_manager import SessionDep
from lightly_studio.models.collection import CollectionTable
from lightly_studio.upload import media_delete

media_delete_router = APIRouter(tags=["media_delete"])


class DeleteImagesRequest(BaseModel):
    """Which images to delete; exactly one selector must be set."""

    sample_ids: list[UUID] | None = Field(None, description="Explicit image sample ids.")
    tag_id: UUID | None = Field(None, description="Every image with this tag, then the tag.")
    delete_all: bool = Field(False, description="Every image of the collection.")


class DeleteImagesView(BaseModel):
    """Outcome of a delete request."""

    deleted_samples: int
    deleted_files: int
    deleted_tag: bool
    deleted_tags: int


@media_delete_router.post(
    "/collections/{collection_id}/images/delete", response_model=DeleteImagesView
)
def delete_images(
    session: SessionDep,
    collection: Annotated[
        CollectionTable,
        FastAPIPath(title="collection Id"),
        Depends(get_and_validate_collection_id),
    ],
    body: DeleteImagesRequest,
) -> DeleteImagesView:
    """Deletes images with their annotations, and the uploaded files behind them."""
    outcome = media_delete.delete_images(
        session=session,
        collection_id=collection.collection_id,
        sample_ids=body.sample_ids,
        tag_id=body.tag_id,
        delete_all=body.delete_all,
    )
    return DeleteImagesView.model_validate(outcome, from_attributes=True)
