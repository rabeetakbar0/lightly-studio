from pathlib import Path
from uuid import UUID, uuid4

import pytest
from PIL import Image
from pytest_mock import MockerFixture
from sqlmodel import Session

from lightly_studio.dataset import env
from lightly_studio.pipeline_datasets import service
from lightly_studio.resolvers import (
    annotation_resolver,
    collection_resolver,
    sample_resolver,
    tag_resolver,
)
from lightly_studio.resolvers.image_filter import ImageFilter
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter
from tests import helpers_resolvers


@pytest.fixture(autouse=True)
def datasets_dir(tmp_path: Path, mocker: MockerFixture) -> Path:
    datasets_dir = tmp_path / "datasets"
    mocker.patch.object(env, "LIGHTLY_STUDIO_DATASETS_DIR", datasets_dir)
    return datasets_dir


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color=(1, 2, 3)).save(path)
    return path


def _collection_with_boxes(session: Session, tmp_path: Path) -> tuple[UUID, UUID, list[UUID]]:
    """Two tagged images with one GateArm box each in 'predictions', one untagged image."""
    collection_id = helpers_resolvers.create_collection(session=session).collection_id
    sample_ids = [
        helpers_resolvers.create_image(
            session=session,
            collection_id=collection_id,
            file_path_abs=str(_png(tmp_path / "src" / f"frame_{i}.png")),
            width=64,
            height=48,
        ).sample_id
        for i in range(3)
    ]
    label = helpers_resolvers.create_annotation_label(
        session=session, root_collection_id=collection_id, label_name="GateArm"
    )
    for sample_id in sample_ids[:2]:
        helpers_resolvers.create_annotation(
            session=session,
            collection_id=collection_id,
            sample_id=sample_id,
            annotation_label_id=label.annotation_label_id,
            annotation_data={"x": 5, "y": 5, "width": 20, "height": 10},
            annotation_collection_name="predictions",
        )
    tag = helpers_resolvers.create_tag(
        session=session, collection_id=collection_id, tag_name="clip"
    )
    tag_resolver.add_sample_ids_to_tag_id(
        session=session, tag_id=tag.tag_id, sample_ids=sample_ids[:2]
    )
    source_id = collection_resolver.get_by_name(
        session=session, name="predictions", parent_collection_id=collection_id
    )
    assert source_id is not None
    return collection_id, source_id, sample_ids


def test_export_dataset(db_session: Session, tmp_path: Path, datasets_dir: Path) -> None:
    collection_id, source_id, _ = _collection_with_boxes(db_session, tmp_path)

    summary = service.export_dataset(
        session=db_session,
        collection_id=collection_id,
        name="data1",
        annotation_collection_id=source_id,
        image_filter=None,
        val_fraction=0.0,
        split_by="tag",
    )

    assert summary.name == "data1"
    assert summary.classes == ["GateArm"]
    assert summary.annotation_source == "predictions"
    assert summary.splits["train2017"].images == 3
    assert summary.splits["train2017"].boxes == 2
    assert (datasets_dir / "data1" / "train2017" / "frame_0.png").exists()


def test_export_dataset__filter_by_tag(db_session: Session, tmp_path: Path) -> None:
    collection_id, source_id, _ = _collection_with_boxes(db_session, tmp_path)
    tag_id = tag_resolver.get_by_name(
        session=db_session, tag_name="clip", collection_id=collection_id
    ).tag_id  # type: ignore[union-attr]

    summary = service.export_dataset(
        session=db_session,
        collection_id=collection_id,
        name="tagged",
        annotation_collection_id=source_id,
        image_filter=ImageFilter(sample_filter=SampleFilter(tag_ids=[tag_id])),
        val_fraction=0.0,
        split_by="tag",
    )
    assert summary.splits["train2017"].images == 2


def test_export_dataset__wrong_source(db_session: Session, tmp_path: Path) -> None:
    collection_id, _, _ = _collection_with_boxes(db_session, tmp_path)
    with pytest.raises(ValueError, match=r"annotation source"):
        service.export_dataset(
            session=db_session,
            collection_id=collection_id,
            name="x",
            annotation_collection_id=uuid4(),
            image_filter=None,
            val_fraction=0.0,
            split_by="tag",
        )


def test_preview_dataset(db_session: Session, tmp_path: Path) -> None:
    collection_id, source_id, _ = _collection_with_boxes(db_session, tmp_path)
    service.export_dataset(
        session=db_session,
        collection_id=collection_id,
        name="data1",
        annotation_collection_id=source_id,
        image_filter=None,
        val_fraction=0.34,
        split_by="image",
    )

    preview_id = service.preview_dataset(session=db_session, name="data1")

    assert preview_id != collection_id
    assert sample_resolver.count_by_collection_id(session=db_session, collection_id=preview_id) == 3
    boxes = annotation_resolver.get_all_by_collection_name(
        session=db_session, collection_name="annotation", parent_collection_id=preview_id
    ).annotations
    assert len(boxes) == 2
    tags = {
        t.name
        for t in tag_resolver.get_all_by_collection_id(session=db_session, collection_id=preview_id)
    }
    assert tags == {"train2017", "val2017"}
    # A second preview reuses the Studio dataset instead of ingesting again.
    assert service.preview_dataset(session=db_session, name="data1") == preview_id


def test_preview_dataset__missing(db_session: Session) -> None:
    with pytest.raises(ValueError, match=r"does not exist"):
        service.preview_dataset(session=db_session, name="nope")
