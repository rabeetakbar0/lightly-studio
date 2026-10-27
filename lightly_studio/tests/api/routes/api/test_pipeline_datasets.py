from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pytest_mock import MockerFixture
from sqlmodel import Session

from lightly_studio.api.routes.api.status import (
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_CREATED,
    HTTP_STATUS_OK,
)
from lightly_studio.dataset import env
from lightly_studio.resolvers import collection_resolver
from tests import helpers_resolvers


@pytest.fixture(autouse=True)
def datasets_dir(tmp_path: Path, mocker: MockerFixture) -> Path:
    datasets_dir = tmp_path / "datasets"
    mocker.patch.object(env, "LIGHTLY_STUDIO_DATASETS_DIR", datasets_dir)
    return datasets_dir


def _collection(session: Session, tmp_path: Path, name: str) -> tuple[str, str]:
    collection_id = helpers_resolvers.create_collection(
        session=session, collection_name=name
    ).collection_id
    # One file name per collection, so two exports can be merged without a name clash.
    path = tmp_path / name / f"{name}.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(path)
    sample_id = helpers_resolvers.create_image(
        session=session, collection_id=collection_id, file_path_abs=str(path), width=8, height=8
    ).sample_id
    label = helpers_resolvers.create_annotation_label(
        session=session, root_collection_id=collection_id
    )
    helpers_resolvers.create_annotation(
        session=session,
        collection_id=collection_id,
        sample_id=sample_id,
        annotation_label_id=label.annotation_label_id,
        annotation_data={"x": 1, "y": 1, "width": 4, "height": 4},
        annotation_collection_name="gt",
    )
    source_id = collection_resolver.get_by_name(
        session=session, name="gt", parent_collection_id=collection_id
    )
    return str(collection_id), str(source_id)


def test_pipeline_datasets_flow(
    db_session: Session, test_client: TestClient, tmp_path: Path
) -> None:
    collection_id, source_id = _collection(db_session, tmp_path, "c1")
    collection_2, source_2 = _collection(db_session, tmp_path / "second", "c2")

    assert test_client.get("/api/pipeline_datasets").json() == []

    response = test_client.post(
        f"/api/collections/{collection_id}/pipeline_datasets/export",
        json={"name": "data1", "annotation_collection_id": source_id, "val_fraction": 0.0},
    )
    assert response.status_code == HTTP_STATUS_CREATED, response.text
    body = response.json()
    assert body["name"] == "data1"
    assert body["images"] == 1
    assert body["boxes"] == 1
    assert body["splits"] == {"train2017": {"images": 1, "boxes": 1}}

    response = test_client.post(
        f"/api/collections/{collection_2}/pipeline_datasets/export",
        json={"name": "data2", "annotation_collection_id": source_2},
    )
    assert response.status_code == HTTP_STATUS_CREATED, response.text

    response = test_client.post(
        "/api/pipeline_datasets/merge", json={"name": "both", "sources": ["data1", "data2"]}
    )
    assert response.status_code == HTTP_STATUS_CREATED, response.text
    assert response.json()["images"] == 2
    assert response.json()["sources"] == ["data1", "data2"]

    names = [d["name"] for d in test_client.get("/api/pipeline_datasets").json()]
    assert set(names) == {"data1", "data2", "both"}

    response = test_client.post("/api/pipeline_datasets/both/preview")
    assert response.status_code == HTTP_STATUS_OK, response.text
    preview_id = response.json()["collection_id"]
    assert test_client.get(f"/api/collections/{preview_id}").json()["total_sample_count"] == 2

    response = test_client.post(
        f"/api/collections/{collection_id}/pipeline_datasets/export",
        json={"name": "data1", "annotation_collection_id": source_id},
    )
    assert response.status_code == HTTP_STATUS_BAD_REQUEST

    assert test_client.delete("/api/pipeline_datasets/data2").status_code == HTTP_STATUS_OK
    assert test_client.delete("/api/pipeline_datasets/data2").status_code == HTTP_STATUS_BAD_REQUEST
