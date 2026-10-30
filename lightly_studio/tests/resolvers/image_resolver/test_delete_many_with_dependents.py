from typing import Any
from uuid import UUID, uuid4

from sqlmodel import Session, col, select

from lightly_studio.models.annotation.annotation_base import AnnotationBaseTable
from lightly_studio.models.caption import CaptionTable
from lightly_studio.models.image import ImageTable
from lightly_studio.models.sample import SampleTable, SampleTagLinkTable
from lightly_studio.resolvers import image_resolver, tag_resolver
from tests import helpers_resolvers


def _count(session: Session, table: Any, ids: list[UUID]) -> int:
    return len(session.exec(select(table).where(col(table.sample_id).in_(ids))).all())


def test_delete_many_with_dependents(db_session: Session) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    # Ids are captured up front: the instances are expired once their rows are gone.
    first = helpers_resolvers.create_image(
        session=db_session, collection_id=collection_id, file_path_abs="/p/first.png"
    ).sample_id
    second = helpers_resolvers.create_image(
        session=db_session, collection_id=collection_id, file_path_abs="/p/second.png"
    ).sample_id
    label = helpers_resolvers.create_annotation_label(
        session=db_session, root_collection_id=collection_id
    )
    annotation = helpers_resolvers.create_annotation(
        session=db_session,
        collection_id=collection_id,
        sample_id=first,
        annotation_label_id=label.annotation_label_id,
        annotation_collection_name="predictions",
    ).sample_id
    tag = helpers_resolvers.create_tag(session=db_session, collection_id=collection_id)
    tag_resolver.add_sample_ids_to_tag_id(
        session=db_session, tag_id=tag.tag_id, sample_ids=[first, second]
    )
    caption = helpers_resolvers.create_caption(
        session=db_session, collection_id=collection_id, parent_sample_id=second
    ).sample_id

    deleted = image_resolver.delete_many_with_dependents(session=db_session, sample_ids=[first])

    assert deleted == 1
    assert _count(db_session, ImageTable, [first]) == 0
    assert _count(db_session, SampleTable, [first, annotation]) == 0
    assert _count(db_session, AnnotationBaseTable, [annotation]) == 0
    assert _count(db_session, SampleTagLinkTable, [first]) == 0
    # The other image, its tag link and caption are untouched; the tag itself stays.
    assert _count(db_session, ImageTable, [second]) == 1
    assert _count(db_session, SampleTagLinkTable, [second]) == 1
    assert _count(db_session, CaptionTable, [caption]) == 1
    assert tag_resolver.get_by_id(session=db_session, tag_id=tag.tag_id) is not None

    deleted = image_resolver.delete_many_with_dependents(
        session=db_session, sample_ids=[second, first]
    )
    assert deleted == 1
    assert _count(db_session, CaptionTable, [caption]) == 0
    assert _count(db_session, SampleTable, [second, caption]) == 0


def test_delete_many_with_dependents__unknown_ids(db_session: Session) -> None:
    assert image_resolver.delete_many_with_dependents(session=db_session, sample_ids=[uuid4()]) == 0
