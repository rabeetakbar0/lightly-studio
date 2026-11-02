"""Exports a collection's images and boxes into a dataset folder, and loads one back."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from uuid import UUID

import numpy as np
from labelformat.formats import COCOObjectDetectionInput
from numpy.typing import NDArray
from sqlmodel import Session, col, select

from lightly_studio.core.image import add_images
from lightly_studio.dataset import env
from lightly_studio.embed import embed_samples
from lightly_studio.models.annotation.annotation_base import AnnotationType
from lightly_studio.models.collection import CollectionCreate, SampleType
from lightly_studio.models.sample import SampleTable, SampleTagLinkTable
from lightly_studio.models.tag import TagTable
from lightly_studio.pipeline_datasets import coco_dataset, val_split
from lightly_studio.pipeline_datasets.coco_dataset import (
    SPLITS,
    BoxRecord,
    DatasetSummary,
    ImagePool,
    ImageRecord,
)
from lightly_studio.pipeline_datasets.val_split import Split
from lightly_studio.resolvers import (
    annotation_label_resolver,
    annotation_resolver,
    collection_embedding_model_resolver,
    collection_resolver,
    grid_filter_sample_ids,
    image_resolver,
    sample_embedding_resolver,
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
    test_fraction: float = 0.0,
    split_strategy: str = val_split.STRATEGY_EMBEDDING,
    groups: int | None = None,
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
        test_fraction: Share of images held out as test.
        split_strategy: "embedding" spreads the hold-outs over similarity groups of the
            image embeddings (computing missing ones first), "random" picks at random.
        groups: Number of similarity groups, None for automatic.

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
    sample_ids = [image.sample_id for image in images]
    tag_by_sample = _first_tag_by_sample(session=session, sample_ids=sample_ids)
    records = [
        ImageRecord(
            path=Path(image.file_path_abs),
            width=image.width,
            height=image.height,
            tag=tag_by_sample.get(image.sample_id, ""),
            sample_id=str(image.sample_id),
        )
        for image in images
    ]
    index_by_sample = {image.sample_id: index for index, image in enumerate(images)}
    split = choose_split(
        session=session,
        collection_id=collection_id,
        sample_ids=sample_ids,
        tags=[record.tag for record in records],
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_by=split_by,
        split_strategy=split_strategy,
        groups=groups,
    )

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
        test_fraction=test_fraction,
        split_by=split_by,
        annotation_source=source.name,
        split=split,
    )


def choose_split(  # noqa: PLR0913
    session: Session,
    collection_id: UUID,
    sample_ids: list[UUID],
    tags: list[str],
    val_fraction: float,
    test_fraction: float,
    split_by: str,
    split_strategy: str,
    groups: int | None,
) -> Split:
    """Picks the hold-outs for the given images, spread over embedding groups when asked.

    Missing embeddings are computed first with the collection's default model. Images that
    still have none are split at random and counted as fallback.
    """
    if split_strategy != val_split.STRATEGY_EMBEDDING or val_fraction + test_fraction <= 0:
        units = val_split.build_units(tags=tags, split_by=split_by, vectors=None)
        return val_split.random_split(
            units=units, val_fraction=val_fraction, test_fraction=test_fraction, split_by=split_by
        )
    vectors = embedding_vectors(session=session, collection_id=collection_id, sample_ids=sample_ids)
    units = val_split.build_units(tags=tags, split_by=split_by, vectors=vectors)
    return val_split.holdout_split(
        units=units,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_by=split_by,
        groups=groups,
    )


def embedding_vectors(
    session: Session, collection_id: UUID, sample_ids: list[UUID]
) -> list[NDArray[np.float32] | None]:
    """One embedding per sample in the given order, None where none could be made.

    Samples without an embedding are embedded with the collection's default model first;
    when no embedder is available they stay None.
    """
    model_id = collection_embedding_model_resolver.get_default_by_collection_id(
        session=session, collection_id=collection_id
    )
    if model_id is None:
        missing = list(sample_ids)
    else:
        unembedded = set(
            image_resolver.get_unembedded_sample_ids(
                session=session, collection_id=collection_id, embedding_model_id=model_id
            )
        )
        missing = [sample_id for sample_id in sample_ids if sample_id in unembedded]
    if missing:
        embed_samples.embed_image_samples(
            session=session, collection_id=collection_id, sample_ids=missing
        )
        model_id = collection_embedding_model_resolver.get_default_by_collection_id(
            session=session, collection_id=collection_id
        )
    if model_id is None:
        return [None] * len(sample_ids)
    rows = sample_embedding_resolver.get_by_sample_ids(
        session=session, sample_ids=list(sample_ids), embedding_model_id=model_id
    )
    by_id = {row.sample_id: np.asarray(row.embedding, dtype=np.float32) for row in rows}
    return [by_id.get(sample_id) for sample_id in sample_ids]


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


def build_test_dataset(  # noqa: PLR0913
    session: Session,
    name: str,
    sources: list[str],
    fraction: float | None,
    count: int | None,
    split_strategy: str = val_split.STRATEGY_EMBEDDING,
    groups: int | None = None,
) -> DatasetSummary:
    """Draws a test-only dataset from all images of the given datasets.

    The images of every split of the sources are pooled (each file name once) and a share
    of them, spread over similarity groups when embeddings are available, becomes the
    test2017 split of a new dataset.

    Args:
        session: The database session, used to read or compute the embeddings.
        name: Folder name of the new dataset.
        sources: The datasets to draw from.
        fraction: Share of the pooled images to take; or
        count: Number of images to take, converted to a fraction of the pool.
        split_strategy: "embedding" or "random".
        groups: Number of similarity groups, None for automatic.

    Raises:
        ValueError: On a bad name, a missing source, neither or both of fraction and count,
            or a share that leaves no image.
    """
    if (fraction is None) == (count is None):
        raise ValueError("Give either a fraction or a count of test images.")
    pool = coco_dataset.load_pool(datasets_dir=env.LIGHTLY_STUDIO_DATASETS_DIR, sources=sources)
    if count is not None:
        if count < 1 or count >= len(pool.images):
            raise ValueError(
                f"The count must be between 1 and {len(pool.images) - 1} for these datasets."
            )
        fraction = count / len(pool.images)
    assert fraction is not None
    if not 0 < fraction < 1:
        raise ValueError("The fraction must be above 0 and below 1.")

    tags = [str(entry.image.get("recording", "")) for entry in pool.images]
    if split_strategy == val_split.STRATEGY_EMBEDDING:
        vectors = _pool_vectors(session=session, pool=pool)
        units = val_split.build_units(tags=tags, split_by=val_split.SPLIT_BY_IMAGE, vectors=vectors)
        split = val_split.holdout_split(
            units=units,
            val_fraction=0.0,
            test_fraction=fraction,
            split_by=val_split.SPLIT_BY_IMAGE,
            groups=groups,
        )
    else:
        units = val_split.build_units(tags=tags, split_by=val_split.SPLIT_BY_IMAGE, vectors=None)
        split = val_split.random_split(
            units=units, val_fraction=0.0, test_fraction=fraction, split_by=val_split.SPLIT_BY_IMAGE
        )
    return coco_dataset.write_test_dataset(
        datasets_dir=env.LIGHTLY_STUDIO_DATASETS_DIR,
        name=name,
        pool=pool,
        test_indices=sorted(split.assignment),
        split=split.info,
    )


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


def _pool_vectors(session: Session, pool: ImagePool) -> list[NDArray[np.float32] | None]:
    """Embeddings for the pool's images through their recorded sample ids, per collection."""
    vectors: list[NDArray[np.float32] | None] = [None] * len(pool.images)
    index_by_sample: dict[UUID, int] = {}
    for index, entry in enumerate(pool.images):
        try:
            index_by_sample[UUID(entry.sample_id)] = index
        except ValueError:
            continue
    if not index_by_sample:
        return vectors
    by_collection: dict[UUID, list[UUID]] = defaultdict(list)
    for sample_id, collection_id in session.exec(
        select(SampleTable.sample_id, SampleTable.collection_id).where(
            col(SampleTable.sample_id).in_(list(index_by_sample))
        )
    ).all():
        by_collection[collection_id].append(sample_id)
    for collection_id, sample_ids in by_collection.items():
        found = embedding_vectors(
            session=session, collection_id=collection_id, sample_ids=sample_ids
        )
        for sample_id, vector in zip(sample_ids, found):
            vectors[index_by_sample[sample_id]] = vector
    return vectors


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
