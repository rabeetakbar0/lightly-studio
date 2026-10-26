"""Deletes image samples together with everything attached to them."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlmodel import Session, col, delete, select

from lightly_studio.models.annotation.annotation_base import AnnotationBaseTable
from lightly_studio.models.annotation_collection_coverage import (
    AnnotationCollectionCoverageTable,
)
from lightly_studio.models.caption import CaptionTable
from lightly_studio.models.evaluation_sample_metric import EvaluationSampleMetricTable
from lightly_studio.models.group import SampleGroupLinkTable
from lightly_studio.models.image import ImageTable
from lightly_studio.models.metadata import SampleMetadataTable
from lightly_studio.models.sample import SampleTable, SampleTagLinkTable
from lightly_studio.models.sample_embedding import SampleEmbeddingTable
from lightly_studio.models.sequence import SampleSequenceLinkTable
from lightly_studio.models.temporal_span import TemporalSpanTable
from lightly_studio.resolvers import annotation_resolver
from lightly_studio.utils import batching

_BATCH_SIZE = 500


def delete_many_with_dependents(session: Session, sample_ids: Sequence[UUID]) -> int:
    """Deletes image samples, their annotations, captions, tags, embeddings and metadata.

    Rows are deleted child first, with a commit after each foreign key level, because
    DuckDB rejects the delete of a parent row in the transaction that deleted its children.
    The delete is therefore not atomic. IDs that do not exist are ignored.

    Args:
        session: The database session.
        sample_ids: The image sample IDs to delete.

    Returns:
        The number of image samples deleted.
    """
    deleted = 0
    for batch in batching.batched(sample_ids, batch_size=_BATCH_SIZE):
        image_ids = list(
            session.exec(select(ImageTable.sample_id).where(col(ImageTable.sample_id).in_(batch)))
        )
        if not image_ids:
            continue
        _delete_annotations(session=session, parent_sample_ids=image_ids)
        caption_ids = list(
            session.exec(
                select(CaptionTable.sample_id).where(
                    col(CaptionTable.parent_sample_id).in_(image_ids)
                )
            )
        )
        all_ids = image_ids + caption_ids
        _delete_attachments(session=session, sample_ids=all_ids, image_ids=image_ids)
        session.exec(delete(CaptionTable).where(col(CaptionTable.sample_id).in_(caption_ids)))
        session.commit()
        session.exec(delete(ImageTable).where(col(ImageTable.sample_id).in_(image_ids)))
        session.commit()
        session.exec(delete(SampleTable).where(col(SampleTable.sample_id).in_(all_ids)))
        session.commit()
        # Bulk deletes bypass the identity map; drop cached instances of the deleted rows.
        session.expire_all()
        deleted += len(image_ids)
    return deleted


def _delete_annotations(session: Session, parent_sample_ids: list[UUID]) -> None:
    """Deletes the annotations drawn on the images, through the annotation resolver."""
    annotation_ids = list(
        session.exec(
            select(AnnotationBaseTable.sample_id).where(
                col(AnnotationBaseTable.parent_sample_id).in_(parent_sample_ids)
            )
        )
    )
    if annotation_ids:
        annotation_resolver.delete_annotations_by_ids(
            session=session, annotation_ids=annotation_ids, delete_samples=True
        )


def _delete_attachments(session: Session, sample_ids: list[UUID], image_ids: list[UUID]) -> None:
    """Deletes every row that references the samples, except the samples themselves."""
    for table, column in (
        (EvaluationSampleMetricTable, EvaluationSampleMetricTable.sample_id),
        (SampleTagLinkTable, SampleTagLinkTable.sample_id),
        (SampleGroupLinkTable, SampleGroupLinkTable.sample_id),
        (SampleSequenceLinkTable, SampleSequenceLinkTable.sample_id),
        (SampleEmbeddingTable, SampleEmbeddingTable.sample_id),
        (SampleMetadataTable, SampleMetadataTable.sample_id),
        (TemporalSpanTable, TemporalSpanTable.sample_id),
    ):
        session.exec(delete(table).where(col(column).in_(sample_ids)))
    session.exec(
        delete(AnnotationCollectionCoverageTable).where(
            col(AnnotationCollectionCoverageTable.parent_sample_id).in_(image_ids)
        )
    )
