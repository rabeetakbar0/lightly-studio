from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from pytest_mock import MockerFixture
from sqlmodel import Session

from lightly_studio.dataset import env
from lightly_studio.resolvers import image_resolver, tag_resolver
from lightly_studio.upload import media_delete
from tests import helpers_resolvers


@pytest.fixture(autouse=True)
def upload_dir(tmp_path: Path, mocker: MockerFixture) -> Path:
    upload_dir = tmp_path / "uploads"
    mocker.patch.object(env, "LIGHTLY_STUDIO_UPLOAD_DIR", upload_dir)
    return upload_dir


def _image(session: Session, collection_id: UUID, path: Path) -> UUID:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return helpers_resolvers.create_image(
        session=session, collection_id=collection_id, file_path_abs=str(path)
    ).sample_id


def test_delete_images__by_tag(db_session: Session, upload_dir: Path) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    frames = upload_dir / str(collection_id) / "frames" / "clip"
    tagged = [_image(db_session, collection_id, frames / f"clip_f{i}.jpg") for i in range(2)]
    kept_upload = _image(
        db_session, collection_id, upload_dir / str(collection_id) / "images" / "k.png"
    )
    tag = helpers_resolvers.create_tag(
        session=db_session, collection_id=collection_id, tag_name="clip"
    )
    tag_resolver.add_sample_ids_to_tag_id(session=db_session, tag_id=tag.tag_id, sample_ids=tagged)

    outcome = media_delete.delete_images(
        session=db_session, collection_id=collection_id, tag_id=tag.tag_id
    )

    assert outcome.deleted_samples == 2
    assert outcome.deleted_files == 2
    assert outcome.deleted_tag is True
    assert not frames.exists()  # The emptied frame folder is pruned.
    assert image_resolver.get_sample_ids(session=db_session, collection_id=collection_id) == {
        kept_upload
    }
    assert tag_resolver.get_by_id(session=db_session, tag_id=tag.tag_id) is None


def test_delete_images__all_keeps_outside_files(
    db_session: Session, upload_dir: Path, tmp_path: Path
) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    _image(db_session, collection_id, upload_dir / str(collection_id) / "images" / "a.png")
    outside = tmp_path / "elsewhere" / "b.png"
    _image(db_session, collection_id, outside)

    helpers_resolvers.create_tag(session=db_session, collection_id=collection_id, tag_name="stale")

    outcome = media_delete.delete_images(
        session=db_session, collection_id=collection_id, delete_all=True
    )

    assert outcome.deleted_samples == 2
    assert outcome.deleted_files == 1
    assert outcome.deleted_tags == 1
    assert (
        tag_resolver.get_all_by_collection_id(session=db_session, collection_id=collection_id) == []
    )
    assert outside.exists()
    assert not (upload_dir / str(collection_id)).exists()
    assert image_resolver.get_sample_ids(session=db_session, collection_id=collection_id) == set()


def test_delete_images__explicit_ids_limited_to_collection(
    db_session: Session, tmp_path: Path
) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    other_id = helpers_resolvers.create_collection(session=db_session).collection_id
    mine = _image(db_session, collection_id, tmp_path / "mine.png")
    theirs = _image(db_session, other_id, tmp_path / "theirs.png")

    outcome = media_delete.delete_images(
        session=db_session, collection_id=collection_id, sample_ids=[mine, theirs]
    )

    assert outcome.deleted_samples == 1
    assert image_resolver.get_sample_ids(session=db_session, collection_id=other_id) == {theirs}


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"delete_all": True, "tag_id": uuid4()}, {"sample_ids": [], "delete_all": True}],
)
def test_delete_images__selector_required(db_session: Session, kwargs: dict[str, Any]) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    with pytest.raises(ValueError, match=r"exactly one"):
        media_delete.delete_images(session=db_session, collection_id=collection_id, **kwargs)


def test_delete_images__unknown_tag(db_session: Session) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    with pytest.raises(ValueError, match=r"does not exist"):
        media_delete.delete_images(session=db_session, collection_id=collection_id, tag_id=uuid4())
