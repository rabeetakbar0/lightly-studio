import io
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import pytest
from PIL import Image
from pytest_mock import MockerFixture
from sqlmodel import Session

from lightly_studio.dataset import env
from lightly_studio.embed import embed_samples
from lightly_studio.resolvers import image_resolver, tag_resolver
from lightly_studio.upload import media_upload
from lightly_studio.upload.frame_selection import FrameSelectionConfig
from tests import helpers_resolvers


@pytest.fixture(autouse=True)
def upload_dir(tmp_path: Path, mocker: MockerFixture) -> Path:
    upload_dir = tmp_path / "uploads"
    mocker.patch.object(env, "LIGHTLY_STUDIO_UPLOAD_DIR", upload_dir)
    return upload_dir


def _png_bytes(width: int = 8, height: int = 6) -> io.BytesIO:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _video_bytes(tmp_path: Path, seconds: int = 4, fps: int = 5) -> io.BytesIO:
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (32, 24))
    for _ in range(seconds * fps):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    return io.BytesIO(path.read_bytes())


def _image_count(session: Session, collection_id: UUID) -> int:
    result = image_resolver.get_all_by_collection_id(session=session, collection_id=collection_id)
    return result.total_count


def test_add_uploaded_images(db_session: Session, upload_dir: Path) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    outcome = media_upload.add_uploaded_images(
        session=db_session,
        collection_id=collection_id,
        files=[("a.png", _png_bytes()), ("b.png", _png_bytes(width=4, height=4))],
        embed=False,
    )
    assert len(outcome.added_sample_ids) == 2
    assert outcome.skipped == 0
    assert outcome.embedded is False
    assert outcome.warnings == []
    assert sorted(p.name for p in (upload_dir / str(collection_id) / "images").iterdir()) == [
        "a.png",
        "b.png",
    ]
    assert _image_count(session=db_session, collection_id=collection_id) == 2


def test_add_uploaded_images__unsupported_and_broken_are_skipped(db_session: Session) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    outcome = media_upload.add_uploaded_images(
        session=db_session,
        collection_id=collection_id,
        files=[
            ("ok.png", _png_bytes()),
            ("notes.txt", io.BytesIO(b"text")),
            ("broken.jpg", io.BytesIO(b"not an image")),
        ],
        embed=False,
    )
    assert len(outcome.added_sample_ids) == 1
    assert outcome.skipped == 2
    assert any("notes.txt" in w for w in outcome.warnings)


def test_add_uploaded_images__no_supported_file(db_session: Session) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    with pytest.raises(ValueError, match=r"supported image"):
        media_upload.add_uploaded_images(
            session=db_session,
            collection_id=collection_id,
            files=[("notes.txt", io.BytesIO(b"text"))],
            embed=False,
        )


def test_add_uploaded_images__embedding_failure_is_a_warning(
    db_session: Session, mocker: MockerFixture
) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    mocker.patch.object(embed_samples, "embed_image_samples", side_effect=RuntimeError("no model"))
    outcome = media_upload.add_uploaded_images(
        session=db_session,
        collection_id=collection_id,
        files=[("a.png", _png_bytes())],
        embed=True,
    )
    assert len(outcome.added_sample_ids) == 1
    assert outcome.embedded is False
    assert "no model" in outcome.warnings[0]


def test_add_uploaded_videos(db_session: Session, tmp_path: Path, upload_dir: Path) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    # 4 s video, 1 s edge windows at 1 s steps, 2 middle frames: 0,1 | 1.67,2.33 | 3,4 s.
    # 4 s is past the last frame and is skipped, so 5 frames remain.
    config = FrameSelectionConfig(
        edge_window_seconds=1.0, edge_interval_seconds=1.0, middle_frames=2, max_frames=100
    )
    outcome = media_upload.add_uploaded_videos(
        session=db_session,
        collection_id=collection_id,
        files=[("gate.mp4", _video_bytes(tmp_path=tmp_path))],
        config=config,
        embed=False,
    )
    assert len(outcome.videos) == 1
    video = outcome.videos[0]
    assert video.file_name == "gate.mp4"
    assert video.duration_seconds == pytest.approx(4.0)
    assert video.frames_extracted == 5
    assert video.frames_added == 5
    assert len(outcome.added_sample_ids) == 5
    assert (upload_dir / str(collection_id) / "videos" / "gate.mp4").exists()
    assert len(list((upload_dir / str(collection_id) / "frames" / "gate").iterdir())) == 5
    assert _image_count(session=db_session, collection_id=collection_id) == 5


def test_add_uploaded_videos__max_frames(db_session: Session, tmp_path: Path) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    config = FrameSelectionConfig(
        edge_window_seconds=1.0, edge_interval_seconds=0.2, middle_frames=10, max_frames=3
    )
    outcome = media_upload.add_uploaded_videos(
        session=db_session,
        collection_id=collection_id,
        files=[("gate.mp4", _video_bytes(tmp_path=tmp_path))],
        config=config,
        embed=False,
    )
    assert outcome.videos[0].frames_extracted <= 3


def test_add_uploaded_videos__undecodable(db_session: Session) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    with pytest.raises(ValueError, match=r"decoded as a video"):
        media_upload.add_uploaded_videos(
            session=db_session,
            collection_id=collection_id,
            files=[("bad.mp4", io.BytesIO(b"garbage"))],
            config=FrameSelectionConfig(),
            embed=False,
        )


def test_save_stream__unique_and_safe_names(tmp_path: Path) -> None:
    first = media_upload.save_stream(
        stream=io.BytesIO(b"1"), file_name="../weird name!.png", target_dir=tmp_path
    )
    second = media_upload.save_stream(
        stream=io.BytesIO(b"2"), file_name="../weird name!.png", target_dir=tmp_path
    )
    assert first.name == "weird_name_.png"
    assert second.name == "weird_name__1.png"
    assert first.read_bytes() == b"1"
    assert second.read_bytes() == b"2"


def test_add_uploaded_images__tag(db_session: Session) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    outcome = media_upload.add_uploaded_images(
        session=db_session,
        collection_id=collection_id,
        files=[("a.png", _png_bytes()), ("b.png", _png_bytes())],
        embed=False,
        tag="site-a",
    )
    tag = tag_resolver.get_by_name(
        session=db_session, tag_name="site-a", collection_id=collection_id
    )
    assert tag is not None
    assert set(tag_resolver.get_sample_ids_by_tag_id(session=db_session, tag_id=tag.tag_id)) == set(
        outcome.added_sample_ids
    )


def test_add_uploaded_videos__default_tag_is_video_name(
    db_session: Session, tmp_path: Path
) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    config = FrameSelectionConfig(
        edge_window_seconds=1.0, edge_interval_seconds=1.0, middle_frames=0
    )
    outcome = media_upload.add_uploaded_videos(
        session=db_session,
        collection_id=collection_id,
        files=[("gate cam.mp4", _video_bytes(tmp_path=tmp_path))],
        config=config,
        embed=False,
    )
    assert outcome.videos[0].tag == "gate cam"
    tag = tag_resolver.get_by_name(
        session=db_session, tag_name="gate cam", collection_id=collection_id
    )
    assert tag is not None
    assert (
        len(tag_resolver.get_sample_ids_by_tag_id(session=db_session, tag_id=tag.tag_id))
        == outcome.videos[0].frames_added
    )


def test_add_uploaded_videos__explicit_empty_tag(db_session: Session, tmp_path: Path) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    outcome = media_upload.add_uploaded_videos(
        session=db_session,
        collection_id=collection_id,
        files=[("gate.mp4", _video_bytes(tmp_path=tmp_path))],
        config=FrameSelectionConfig(edge_window_seconds=1.0, middle_frames=0),
        embed=False,
        tags=[""],
    )
    assert outcome.videos[0].tag == ""
    assert (
        tag_resolver.get_by_name(session=db_session, tag_name="gate", collection_id=collection_id)
        is None
    )
