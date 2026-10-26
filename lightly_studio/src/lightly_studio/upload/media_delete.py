"""Removes image samples from a collection, together with the files the GUI uploaded."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from lightly_studio.dataset import env
from lightly_studio.models.collection import CollectionTable, SampleType
from lightly_studio.resolvers import evaluation_run_resolver, image_resolver, tag_resolver

# Delete the uploaded files in chunks, so the database stays ahead of the file system.
_FILE_BATCH_SIZE = 1000


@dataclass
class DeleteOutcome:
    """Summary of one delete request.

    Attributes:
        deleted_samples: Number of image samples removed from the database.
        deleted_files: Number of uploaded files removed from disk.
        deleted_tag: Whether the tag itself was removed.
        deleted_tags: Number of tags removed. With delete_all every tag of the collection goes.
    """

    deleted_samples: int = 0
    deleted_files: int = 0
    deleted_tag: bool = False
    deleted_tags: int = 0


def delete_images(
    session: Session,
    collection_id: UUID,
    sample_ids: list[UUID] | None = None,
    tag_id: UUID | None = None,
    delete_all: bool = False,
) -> DeleteOutcome:
    """Deletes image samples of the collection and the uploaded files behind them.

    Exactly one selector applies: explicit sample ids, every sample carrying a tag, or the
    whole collection. Files are removed only when they live in the upload folder; files
    that were ingested from elsewhere on disk stay where they are.

    Args:
        session: The database session.
        collection_id: The image collection.
        sample_ids: Explicit image sample ids to delete.
        tag_id: Delete every image carrying this tag, then the tag itself.
        delete_all: Delete every image of the collection, all of its tags, and its whole
            upload folder, so the collection is a clean slate.

    Returns:
        The outcome.

    Raises:
        ValueError: If no selector or more than one is given, or the tag does not exist.
    """
    targets = _resolve_targets(
        session=session,
        collection_id=collection_id,
        sample_ids=sample_ids,
        tag_id=tag_id,
        delete_all=delete_all,
    )
    outcome = DeleteOutcome()
    upload_dir = _collection_dir(collection_id=collection_id)
    ordered = sorted(targets)
    for start in range(0, len(ordered), _FILE_BATCH_SIZE):
        chunk = ordered[start : start + _FILE_BATCH_SIZE]
        paths = [
            Path(image.file_path_abs)
            for image in image_resolver.get_many_by_id(session=session, sample_ids=chunk)
        ]
        outcome.deleted_samples += image_resolver.delete_many_with_dependents(
            session=session, sample_ids=chunk
        )
        outcome.deleted_files += _remove_uploaded_files(paths=paths, upload_dir=upload_dir)

    if outcome.deleted_samples:
        _mark_evaluations_stale(session=session, collection_id=collection_id)
    if tag_id is not None:
        outcome.deleted_tag = tag_resolver.delete(session=session, tag_id=tag_id)
        outcome.deleted_tags = int(outcome.deleted_tag)
    if delete_all:
        for tag in tag_resolver.get_all_by_collection_id(
            session=session, collection_id=collection_id
        ):
            outcome.deleted_tags += int(tag_resolver.delete(session=session, tag_id=tag.tag_id))
        if upload_dir.exists():
            # Nothing references the folder any more; the videos and frame folders go too.
            shutil.rmtree(upload_dir)
    return outcome


def _resolve_targets(
    session: Session,
    collection_id: UUID,
    sample_ids: list[UUID] | None,
    tag_id: UUID | None,
    delete_all: bool,
) -> set[UUID]:
    """Turns the one chosen selector into the image sample ids to delete."""
    selectors = sum(
        1 for chosen in (sample_ids is not None, tag_id is not None, delete_all) if chosen
    )
    if selectors != 1:
        raise ValueError("Give exactly one of sample_ids, tag_id or delete_all.")
    if delete_all:
        return image_resolver.get_sample_ids(session=session, collection_id=collection_id)
    if tag_id is not None:
        tag = tag_resolver.get_by_id(session=session, tag_id=tag_id)
        if tag is None or tag.collection_id != collection_id:
            raise ValueError(f"Tag {tag_id} does not exist in this collection.")
        return set(tag_resolver.get_sample_ids_by_tag_id(session=session, tag_id=tag_id))
    assert sample_ids is not None
    in_collection = image_resolver.get_sample_ids(session=session, collection_id=collection_id)
    return set(sample_ids) & in_collection


def _mark_evaluations_stale(session: Session, collection_id: UUID) -> None:
    """Evaluations over the collection's annotation layers no longer match the data."""
    annotation_collections = session.exec(
        select(CollectionTable).where(
            CollectionTable.parent_collection_id == collection_id,
            CollectionTable.sample_type == SampleType.ANNOTATION,
        )
    ).all()
    for annotation_collection in annotation_collections:
        evaluation_run_resolver.mark_stale_by_collection_id(
            session=session, collection_id=annotation_collection.collection_id
        )


def _collection_dir(collection_id: UUID) -> Path:
    return env.LIGHTLY_STUDIO_UPLOAD_DIR.resolve() / str(collection_id)


def _remove_uploaded_files(paths: list[Path], upload_dir: Path) -> int:
    """Unlinks the files inside upload_dir and prunes frame folders that became empty."""
    removed = 0
    parents: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if upload_dir not in resolved.parents or not resolved.is_file():
            continue
        resolved.unlink()
        removed += 1
        parents.add(resolved.parent)
    for parent in parents:
        if parent != upload_dir and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    return removed
