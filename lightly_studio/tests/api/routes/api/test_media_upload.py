import io
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pytest_mock import MockerFixture
from sqlmodel import Session

from lightly_studio.api.routes.api.status import (
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_CREATED,
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_OK,
)
from lightly_studio.dataset import env
from lightly_studio.models.collection import SampleType
from tests import helpers_resolvers


@pytest.fixture(autouse=True)
def upload_dir(tmp_path: Path, mocker: MockerFixture) -> Path:
    upload_dir = tmp_path / "uploads"
    mocker.patch.object(env, "LIGHTLY_STUDIO_UPLOAD_DIR", upload_dir)
    return upload_dir


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_upload_images(db_session: Session, test_client: TestClient) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    response = test_client.post(
        f"/api/collections/{collection_id}/uploads/images",
        files=[
            ("files", ("a.png", _png(), "image/png")),
            ("files", ("b.png", _png(), "image/png")),
        ],
        data={"embed": "false"},
    )
    assert response.status_code == HTTP_STATUS_CREATED, response.text
    body = response.json()
    assert len(body["added_sample_ids"]) == 2
    assert body["skipped"] == 0
    assert body["embedded"] is False
    assert body["videos"] == []


def test_upload_images__collection_not_found(test_client: TestClient) -> None:
    response = test_client.post(
        f"/api/collections/{uuid4()}/uploads/images",
        files=[("files", ("a.png", _png(), "image/png"))],
    )
    assert response.status_code == HTTP_STATUS_NOT_FOUND


def test_upload_images__video_collection_is_rejected(
    db_session: Session, test_client: TestClient
) -> None:
    collection_id = helpers_resolvers.create_collection(
        session=db_session, sample_type=SampleType.VIDEO
    ).collection_id
    response = test_client.post(
        f"/api/collections/{collection_id}/uploads/images",
        files=[("files", ("a.png", _png(), "image/png"))],
        data={"embed": "false"},
    )
    assert response.status_code == HTTP_STATUS_BAD_REQUEST


def test_upload_videos__undecodable(db_session: Session, test_client: TestClient) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    response = test_client.post(
        f"/api/collections/{collection_id}/uploads/videos",
        files=[("files", ("bad.mp4", b"garbage", "video/mp4"))],
        data={"embed": "false", "max_frames": "10"},
    )
    assert response.status_code == HTTP_STATUS_BAD_REQUEST


def test_plan_video_frames(test_client: TestClient) -> None:
    response = test_client.post(
        "/api/uploads/frame_plan",
        json={
            "durations_seconds": [100.0, 10.0],
            "edge_window_seconds": 20.0,
            "edge_interval_seconds": 1.0,
            "middle_frames": 3,
            "max_frames": 1000,
        },
    )
    assert response.status_code == HTTP_STATUS_OK
    # 21 + 21 + 3 for the long video; the short one is all edge at 1 s steps.
    assert response.json() == {"frame_counts": [45, 11], "total_frames": 56}
