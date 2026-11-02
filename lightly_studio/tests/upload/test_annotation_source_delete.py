from uuid import UUID

import pytest
from sqlmodel import Session, col, select

from lightly_studio.models.annotation.annotation_base import AnnotationBaseTable
from lightly_studio.models.annotation_label import AnnotationLabelTable
from lightly_studio.models.evaluation_class_metric import (
    EvaluationClassMetricCreate,
    EvaluationClassMetricTable,
)
from lightly_studio.models.evaluation_run import (
    EvaluationRunCreate,
    EvaluationRunTable,
    EvaluationTaskType,
)
from lightly_studio.models.sample import SampleTable
from lightly_studio.resolvers import (
    collection_resolver,
    evaluation_class_metric_resolver,
    evaluation_run_resolver,
    image_resolver,
)
from lightly_studio.upload import annotation_source_delete, media_delete
from tests import helpers_resolvers


def _setup(session: Session) -> tuple[UUID, UUID, UUID, list[UUID]]:
    """Two images; "annotation" has a GateArm box on both, "pred" a LedLight box on one.

    Returns the collection id, the ids of the "annotation" and "pred" sources, and the images.
    """
    collection = helpers_resolvers.create_collection(session=session)
    collection_id = collection.collection_id
    images = [
        helpers_resolvers.create_image(
            session=session, collection_id=collection_id, file_path_abs=f"/x/img_{i}.png"
        ).sample_id
        for i in range(2)
    ]
    gate = helpers_resolvers.create_annotation_label(
        session=session, root_collection_id=collection_id, label_name="GateArm"
    )
    led = helpers_resolvers.create_annotation_label(
        session=session, root_collection_id=collection_id, label_name="LedLight"
    )
    for image in images:
        helpers_resolvers.create_annotation(
            session=session,
            collection_id=collection_id,
            sample_id=image,
            annotation_label_id=gate.annotation_label_id,
            annotation_collection_name="annotation",
        )
    helpers_resolvers.create_annotation(
        session=session,
        collection_id=collection_id,
        sample_id=images[0],
        annotation_label_id=led.annotation_label_id,
        annotation_collection_name="pred",
    )
    gt_id = collection_resolver.get_by_name(
        session=session, name="annotation", parent_collection_id=collection_id
    )
    pred_id = collection_resolver.get_by_name(
        session=session, name="pred", parent_collection_id=collection_id
    )
    assert gt_id is not None
    assert pred_id is not None
    run = evaluation_run_resolver.create(
        session=session,
        evaluation_run_input=EvaluationRunCreate(
            name="run",
            gt_annotation_collection_id=gt_id,
            pred_annotation_collection_id=pred_id,
            dataset_id=collection.dataset_id,
            task_type=EvaluationTaskType.OBJECT_DETECTION,
        ),
    )
    evaluation_class_metric_resolver.create_many(
        session=session,
        records=[
            EvaluationClassMetricCreate(
                evaluation_run_id=run.id,
                annotation_label_id=led.annotation_label_id,
                metric_name="ap",
                value=0.5,
            )
        ],
    )
    return collection_id, gt_id, pred_id, images


def _label_names(session: Session) -> set[str]:
    return {label.annotation_label_name for label in session.exec(select(AnnotationLabelTable))}


def test_delete_annotation_source(db_session: Session) -> None:
    collection_id, gt_id, pred_id, images = _setup(db_session)

    outcome = annotation_source_delete.delete_annotation_source(
        session=db_session, collection_id=collection_id, annotation_collection_id=pred_id
    )

    assert outcome.deleted_sources == 1
    assert outcome.deleted_annotations == 1
    assert outcome.deleted_evaluations == 1
    assert outcome.deleted_classes == 1  # LedLight was only used by "pred".
    assert collection_resolver.get_by_id(session=db_session, collection_id=pred_id) is None
    assert db_session.exec(select(EvaluationRunTable)).all() == []
    assert db_session.exec(select(EvaluationClassMetricTable)).all() == []
    assert (
        db_session.exec(select(SampleTable).where(SampleTable.collection_id == pred_id)).all() == []
    )
    assert _label_names(db_session) == {"GateArm"}
    # The images and the other source stay.
    assert image_resolver.get_sample_ids(session=db_session, collection_id=collection_id) == set(
        images
    )
    remaining = db_session.exec(
        select(SampleTable.collection_id).join(
            AnnotationBaseTable, col(AnnotationBaseTable.sample_id) == col(SampleTable.sample_id)
        )
    ).all()
    assert remaining == [gt_id, gt_id]


def test_delete_annotation_source__keep_unused_classes(db_session: Session) -> None:
    collection_id, _, pred_id, _ = _setup(db_session)
    outcome = annotation_source_delete.delete_annotation_source(
        session=db_session,
        collection_id=collection_id,
        annotation_collection_id=pred_id,
        delete_unused_classes=False,
    )
    assert outcome.deleted_classes == 0
    assert _label_names(db_session) == {"GateArm", "LedLight"}


def test_delete_annotation_source__wrong_collection(db_session: Session) -> None:
    collection_id, _, _, _ = _setup(db_session)
    with pytest.raises(ValueError, match="does not belong"):
        annotation_source_delete.delete_annotation_source(
            session=db_session, collection_id=collection_id, annotation_collection_id=collection_id
        )


def test_delete_images__all_removes_annotation_sources(db_session: Session) -> None:
    collection_id, gt_id, pred_id, _ = _setup(db_session)

    outcome = media_delete.delete_images(
        session=db_session, collection_id=collection_id, delete_all=True
    )

    assert outcome.deleted_samples == 2
    assert outcome.deleted_sources == 2
    assert outcome.deleted_classes == 2
    assert (
        collection_resolver.get_annotation_collections(
            session=db_session, parent_collection_id=collection_id
        )
        == []
    )
    assert collection_resolver.get_by_id(session=db_session, collection_id=gt_id) is None
    assert collection_resolver.get_by_id(session=db_session, collection_id=pred_id) is None
    assert _label_names(db_session) == set()
    assert db_session.exec(select(EvaluationRunTable)).all() == []
