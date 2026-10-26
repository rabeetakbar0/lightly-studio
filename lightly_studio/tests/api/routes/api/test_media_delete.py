from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from sqlmodel import Session

from lightly_studio.api.routes.api.status import (
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_OK,
)
from lightly_studio.dataset import env
from lightly_studio.resolvers import tag_resolver
from tests import helpers_resolvers


@pytest.fixture(autouse=True)
def upload_dir(tmp_path: Path, mocker: MockerFixture) -> Path:
    upload_dir = tmp_path / "uploads"
    mocker.patch.object(env, "LIGHTLY_STUDIO_UPLOAD_DIR", upload_dir)
    return upload_dir


def test_delete_images(db_session: Session, test_client: TestClient) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    first = helpers_resolvers.create_image(session=db_session, collection_id=collection_id)
    helpers_resolvers.create_image(
        session=db_session, collection_id=collection_id, file_path_abs="/p/second.png"
    )
    tag = helpers_resolvers.create_tag(session=db_session, collection_id=collection_id)
    tag_resolver.add_sample_ids_to_tag_id(
        session=db_session, tag_id=tag.tag_id, sample_ids=[first.sample_id]
    )

    response = test_client.post(
        f"/api/collections/{collection_id}/images/delete", json={"tag_id": str(tag.tag_id)}
    )
    assert response.status_code == HTTP_STATUS_OK, response.text
    assert response.json() == {
        "deleted_samples": 1,
        "deleted_files": 0,
        "deleted_tag": True,
        "deleted_tags": 1,
    }

    response = test_client.post(
        f"/api/collections/{collection_id}/images/delete", json={"delete_all": True}
    )
    assert response.status_code == HTTP_STATUS_OK
    assert response.json()["deleted_samples"] == 1
    assert test_client.get(f"/api/collections/{collection_id}").json()["total_sample_count"] == 0


def test_delete_images__bad_selector(db_session: Session, test_client: TestClient) -> None:
    collection_id = helpers_resolvers.create_collection(session=db_session).collection_id
    response = test_client.post(f"/api/collections/{collection_id}/images/delete", json={})
    assert response.status_code == HTTP_STATUS_BAD_REQUEST


def test_delete_images__collection_not_found(test_client: TestClient) -> None:
    response = test_client.post(
        f"/api/collections/{uuid4()}/images/delete", json={"delete_all": True}
    )
    assert response.status_code == HTTP_STATUS_NOT_FOUND
