"""Exports a collection's images and boxes into a dataset folder, and loads one back."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from uuid import UUID

from labelformat.formats import COCOObjectDetectionInput
from sqlmodel import Session, col, select

from lightly_studio.core.image import add_images
from lightly_studio.dataset import env
from lightly_studio.models.annotation.annotation_base import AnnotationType
from lightly_studio.models.collection import CollectionCreate, SampleType
from lightly_studio.models.sample import SampleTagLinkTable
from lightly_studio.models.tag import TagTable
from lightly_studio.pipeline_datasets import coco_dataset
from lightly_studio.pipeline_datasets.coco_dataset import (
    SPLITS,
    BoxRecord,
    DatasetSummary,
    ImageRecord,
)
from lightly_studio.resolvers import (
    annotation_label_resolver,
    annotation_resolver,
    collection_resolver,
    grid_filter_sample_ids,
    image_resolver,
    sample_resolver,
    tag_resolver,
)
from lightly_studio.resolvers.image_filter import ImageFilter

# Studio datasets created to look at a prepared dataset carry this prefix, so the landing
# page can skip them and open the working dataset instead.
PREVIEW_PREFIX = "preview:"
GROUND_TRUTH_SOURCE = "annotation"


def export_dataset(  # noqa: PLR0913
    session: Session,
    collection_id: UUID,
    name: str,
    annotation_collection_id: UUID,
    image_filter: ImageFilter | None,
    val_fraction: float,
    split_by: str,
) -> DatasetSummary:
    """Writes the images in the current view and their boxes as a new dataset folder.

    Args:
        session: The database session.
        collection_id: The image collection.
        name: Folder name of the new dataset.
        annotation_collection_id: The annotation source whose boxes are exported.
        image_filter: The grid filter; None means the whole collection.
        val_fraction: Share of images held out as validation.
        split_by: "tag" keeps whole tags together, "image" splits single images.

    Returns:
        The summary of the written dataset.

    Raises:
        ValueError: On a bad name, an existing dataset, an unknown annotation source, or
            no images in the view.
    """
    source = collection_resolver.get_by_id(session=session, collection_id=annotation_collection_id)
    if source is None or source.parent_collection_id != collection_id:
        raise ValueError("The annotation source does not belong to this collection.")

    sample_ids = _sample_ids_in_view(
        session=session, collection_id=collection_id, image_filter=image_filter
    )
    images = image_resolver.get_many_by_id(session=session, sample_ids=sample_ids)
    tag_by_sample = _first_tag_by_sample(session=session, sample_ids=sample_ids)
    records = [
        ImageRecord(
            path=Path(image.file_path_abs),
            width=image.width,
            height=image.height,
            tag=tag_by_sample.get(image.sample_id, ""),
        )
        for image in images
    ]
    index_by_sample = {image.sample_id: index for index, image in enumerate(images)}

    annotations = annotation_resolver.get_all_by_collection_id_and_parent_sample_ids(
        session=session,
        parent_sample_ids=sample_ids,
        annotation_collection_id=annotation_collection_id,
        annotation_type=AnnotationType.OBJECT_DETECTION,
    )
    label_ids = sorted({annotation.annotation_label_id for annotation in annotations})
    label_name = {
        label.annotation_label_id: label.annotation_label_name
        for label in annotation_label_resolver.get_by_ids(session=session, ids=label_ids)
    }
    boxes = [
        BoxRecord(
            image_index=index_by_sample[annotation.parent_sample_id],
            class_name=label_name[annotation.annotation_label_id],
            x=annotation.object_detection_details.x,
            y=annotation.object_detection_details.y,
            width=annotation.object_detection_details.width,
            height=annotation.object_detection_details.height,
        )
        for annotation in annotations
        if annotation.object_detection_details is not None
    ]
    return coco_dataset.write_dataset(
        datasets_dir=env.LIGHTLY_STUDIO_DATASETS_DIR,
        name=name,
        images=records,
        boxes=boxes,
        class_names=sorted(label_name.values(), key=str.lower),
        val_fraction=val_fraction,
        split_by=split_by,
        annotation_source=source.name,
    )


def preview_dataset(session: Session, name: str) -> UUID:
    """Loads a dataset folder into Studio as its own dataset and returns the root collection.

    The Studio dataset is named "preview:<name>". When it already holds images, it is
    reused as is. Images get the ground truth annotation source and a tag per split.

    Raises:
        ValueError: If the dataset folder does not exist.
    """
    coco_dataset.validate_name(name=name)
    dataset_dir = env.LIGHTLY_STUDIO_DATASETS_DIR / name
    if coco_dataset.read_summary(dataset_dir=dataset_dir) is None:
        raise ValueError(f"Dataset '{name}' does not exist.")

    studio_name = f"{PREVIEW_PREFIX}{name}"
    collection_id = collection_resolver.get_by_name(
        session=session, name=studio_name, parent_collection_id=None
    )
    if collection_id is None:
        collection_id = collection_resolver.create(
            session=session,
            collection=CollectionCreate(name=studio_name, sample_type=SampleType.IMAGE),
        ).collection_id
    if sample_resolver.count_by_collection_id(session=session, collection_id=collection_id) > 0:
        return collection_id

    for split in SPLITS:
        json_path = dataset_dir / "annotations" / f"instances_{split}.json"
        if not json_path.is_file():
            continue
        created = add_images.load_into_dataset_from_labelformat(
            session=session,
            root_collection_id=collection_id,
            input_labels=COCOObjectDetectionInput(input_file=json_path),
            images_path=dataset_dir / split,
            collection_name=GROUND_TRUTH_SOURCE,
        )
        if created:
            tag = tag_resolver.get_or_create_sample_tag_by_name(
                session=session, collection_id=collection_id, tag_name=split
            )
            tag_resolver.add_sample_ids_to_tag_id(
                session=session, tag_id=tag.tag_id, sample_ids=created
            )
    return collection_id


def list_datasets() -> list[DatasetSummary]:
    """The datasets in the datasets folder, newest first."""
    return coco_dataset.list_datasets(datasets_dir=env.LIGHTLY_STUDIO_DATASETS_DIR)


def merge_datasets(name: str, sources: list[str]) -> DatasetSummary:
    """Merges datasets of the datasets folder into a new one."""
    return coco_dataset.merge_datasets(
        datasets_dir=env.LIGHTLY_STUDIO_DATASETS_DIR, name=name, sources=sources
    )


def delete_dataset(name: str) -> None:
    """Removes a dataset folder. The Studio preview dataset, if any, stays."""
    coco_dataset.delete_dataset(datasets_dir=env.LIGHTLY_STUDIO_DATASETS_DIR, name=name)


def _sample_ids_in_view(
    session: Session, collection_id: UUID, image_filter: ImageFilter | None
) -> list[UUID]:
    if image_filter is None:
        return sorted(image_resolver.get_sample_ids(session=session, collection_id=collection_id))
    query = grid_filter_sample_ids.build_sample_ids_query(
        session=session, collection_id=collection_id, grid_filter=image_filter
    )
    return sorted(session.exec(query).all())


def _first_tag_by_sample(session: Session, sample_ids: list[UUID]) -> dict[UUID, str]:
    """The alphabetically first tag name of each sample, for samples that have one."""
    names: dict[UUID, list[str]] = defaultdict(list)
    for sample_id, tag_name in session.exec(
        select(SampleTagLinkTable.sample_id, TagTable.name)
        .join(TagTable, col(TagTable.tag_id) == col(SampleTagLinkTable.tag_id))
        .where(col(SampleTagLinkTable.sample_id).in_(sample_ids))
    ).all():
        if sample_id is not None:
            names[sample_id].append(tag_name)
    return {sample_id: min(tags) for sample_id, tags in names.items()}
