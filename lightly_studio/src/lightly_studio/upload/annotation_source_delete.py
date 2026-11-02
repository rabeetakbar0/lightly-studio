"""Removes an annotation source of a collection with everything recorded under it."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session, col, delete, or_, select

from lightly_studio.models.annotation.annotation_base import AnnotationBaseTable
from lightly_studio.models.annotation_collection_coverage import (
    AnnotationCollectionCoverageTable,
)
from lightly_studio.models.annotation_label import AnnotationLabelTable
from lightly_studio.models.collection import CollectionTable, SampleType
from lightly_studio.models.collection_embedding_model import CollectionEmbeddingModelTable
from lightly_studio.models.evaluation_class_metric import EvaluationClassMetricTable
from lightly_studio.models.evaluation_run import EvaluationRunTable
from lightly_studio.models.export_job import ExportJobTable
from lightly_studio.models.sample import SampleTable, SampleTagLinkTable
from lightly_studio.models.sample_embedding import SampleEmbeddingTable
from lightly_studio.models.tag import TagTable
from lightly_studio.resolvers import (
    annotation_resolver,
    collection_resolver,
    evaluation_run_resolver,
    tag_resolver,
)
from lightly_studio.utils import batching

_BATCH_SIZE = 500


@dataclass
class SourceDeleteOutcome:
    """Summary of removing annotation sources.

    Attributes:
        deleted_sources: Annotation sources removed.
        deleted_annotations: Annotations removed with them.
        deleted_evaluations: Evaluation runs that compared against a removed source.
        deleted_classes: Annotation classes that no remaining annotation uses.
    """

    deleted_sources: int = 0
    deleted_annotations: int = 0
    deleted_evaluations: int = 0
    deleted_classes: int = 0


def delete_annotation_source(
    session: Session,
    collection_id: UUID,
    annotation_collection_id: UUID,
    delete_unused_classes: bool = True,
) -> SourceDeleteOutcome:
    """Deletes one annotation source of the collection and everything under it.

    Removes its annotations and their samples, tags, embeddings and coverage rows, the
    evaluation runs that used it as ground truth or prediction, its tags, and the source
    itself. The images are not touched.

    Args:
        session: The database session.
        collection_id: The image collection the source belongs to.
        annotation_collection_id: The annotation source to delete.
        delete_unused_classes: Also delete the annotation classes of the dataset that no
            remaining annotation uses.

    Returns:
        The outcome.

    Raises:
        ValueError: If the source is not an annotation source of this collection.
    """
    source = collection_resolver.get_by_id(session=session, collection_id=annotation_collection_id)
    if (
        source is None
        or source.parent_collection_id != collection_id
        or source.sample_type != SampleType.ANNOTATION
    ):
        raise ValueError("The annotation source does not belong to this collection.")
    outcome = SourceDeleteOutcome()
    _delete_source(session=session, source=source, outcome=outcome)
    if delete_unused_classes:
        outcome.deleted_classes = _delete_unused_classes(
            session=session, dataset_id=source.dataset_id
        )
    return outcome


def delete_all_annotation_sources(session: Session, collection_id: UUID) -> SourceDeleteOutcome:
    """Deletes every annotation source of the collection and every unused annotation class."""
    outcome = SourceDeleteOutcome()
    collection = collection_resolver.get_by_id(session=session, collection_id=collection_id)
    if collection is None:
        return outcome
    for source in collection_resolver.get_annotation_collections(
        session=session, parent_collection_id=collection_id
    ):
        _delete_source(session=session, source=source, outcome=outcome)
    outcome.deleted_classes = _delete_unused_classes(
        session=session, dataset_id=collection.dataset_id
    )
    return outcome


def _delete_source(session: Session, source: CollectionTable, outcome: SourceDeleteOutcome) -> None:
    source_id = source.collection_id
    outcome.deleted_evaluations += _delete_evaluations(session=session, source_id=source_id)

    # An annotation belongs to the source through its sample's collection.
    annotation_ids = list(
        session.exec(
            select(AnnotationBaseTable.sample_id)
            .join(SampleTable, col(SampleTable.sample_id) == col(AnnotationBaseTable.sample_id))
            .where(col(SampleTable.collection_id) == source_id)
        )
    )
    for batch in batching.batched(annotation_ids, batch_size=_BATCH_SIZE):
        outcome.deleted_annotations += annotation_resolver.delete_annotations_by_ids(
            session=session, annotation_ids=list(batch), delete_samples=True
        )

    # Samples of the source that are not annotations (none are expected) would block it.
    leftover = list(
        session.exec(
            select(SampleTable.sample_id).where(col(SampleTable.collection_id) == source_id)
        )
    )
    for batch in batching.batched(leftover, batch_size=_BATCH_SIZE):
        ids = list(batch)
        session.exec(delete(SampleTagLinkTable).where(col(SampleTagLinkTable.sample_id).in_(ids)))
        session.exec(
            delete(SampleEmbeddingTable).where(col(SampleEmbeddingTable.sample_id).in_(ids))
        )
        session.commit()
        session.exec(delete(SampleTable).where(col(SampleTable.sample_id).in_(ids)))
        session.commit()

    for tag in session.exec(select(TagTable).where(col(TagTable.collection_id) == source_id)).all():
        tag_resolver.delete(session=session, tag_id=tag.tag_id)
    for table, column in (
        (
            AnnotationCollectionCoverageTable,
            AnnotationCollectionCoverageTable.annotation_collection_id,
        ),
        (CollectionEmbeddingModelTable, CollectionEmbeddingModelTable.collection_id),
        (ExportJobTable, ExportJobTable.collection_id),
    ):
        session.exec(delete(table).where(col(column) == source_id))
    session.commit()
    session.expire_all()
    collection_resolver.delete(session=session, collection_id=source_id)
    outcome.deleted_sources += 1


def _delete_evaluations(session: Session, source_id: UUID) -> int:
    """Deletes the evaluation runs that compared against the source, with all their metrics."""
    runs = session.exec(
        select(EvaluationRunTable).where(
            or_(
                col(EvaluationRunTable.gt_annotation_collection_id) == source_id,
                col(EvaluationRunTable.pred_annotation_collection_id) == source_id,
            )
        )
    ).all()
    for run in runs:
        # The run resolver leaves the class metrics, which reference the run.
        session.exec(
            delete(EvaluationClassMetricTable).where(
                col(EvaluationClassMetricTable.evaluation_run_id) == run.id
            )
        )
        evaluation_run_resolver.delete(session=session, run=run)
    return len(runs)


def _delete_unused_classes(session: Session, dataset_id: UUID) -> int:
    """Deletes the dataset's annotation classes that no annotation uses any more."""
    used = select(AnnotationBaseTable.annotation_label_id).distinct()
    unused = session.exec(
        select(AnnotationLabelTable).where(
            col(AnnotationLabelTable.dataset_id) == dataset_id,
            col(AnnotationLabelTable.annotation_label_id).not_in(used),
        )
    ).all()
    for label in unused:
        session.delete(label)
    session.commit()
    return len(unused)
