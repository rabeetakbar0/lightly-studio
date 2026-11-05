import os
from pathlib import Path

from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from lightly_studio.api.routes.api.status import (
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_CREATED,
    HTTP_STATUS_OK,
)
from tests.video_tests import helpers


def test_get_video_test_options(
    test_client: TestClient, tmp_path: Path, mocker: MockerFixture
) -> None:
    helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
    body = test_client.get("/api/video_tests/options").json()
    assert body == {
        "ready": True,
        "problems": [],
        "testapp_dir": str(tmp_path / "TestApp"),
        "uploads_dir": str(tmp_path / "TestApp" / "uploads"),
        "results_dir": str(tmp_path / "TestApp" / "uploads" / "results"),
        "models": [{"name": "model_a", "classes": ["GateArm", "LedLight"]}],
    }


def test_video_tests_flow(test_client: TestClient, tmp_path: Path, mocker: MockerFixture) -> None:
    runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
    response = test_client.post(
        "/api/video_tests",
        data={"model": "model_a"},
        files=[("files", ("clip.mp4", b"video", "video/mp4"))],
    )
    assert response.status_code == HTTP_STATUS_CREATED
    name = response.json()["name"]
    assert response.json()["videos"][0]["file_name"] == "clip.mp4"
    helpers.wait_until_done(runner, name)

    runs = test_client.get("/api/video_tests").json()
    assert [run["name"] for run in runs] == [name]
    run = test_client.get(f"/api/video_tests/{name}").json()
    assert run["status"] == "finished"
    assert run["progress"] is None
    output = tmp_path / "TestApp" / "uploads" / "results" / name / "clip_processed.mp4"
    assert run["videos"][0]["output_path"] == str(output)
    assert run["videos"][0]["status"] == "done"

    log = test_client.get(f"/api/video_tests/{name}/log", params={"lines": 2}).json()
    assert "Batch finished" in log["text"]

    response = test_client.post(f"/api/video_tests/{name}/stop")
    assert response.status_code == HTTP_STATUS_BAD_REQUEST


def test_video_tests_progress_while_running(
    test_client: TestClient, tmp_path: Path, mocker: MockerFixture
) -> None:
    runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
    mocker.patch.dict(os.environ, {"STUB_SLEEP_SECONDS": "30"})
    response = test_client.post(
        "/api/video_tests",
        data={"model": "model_a"},
        files=[("files", ("clip.mp4", b"video", "video/mp4"))],
    )
    name = response.json()["name"]
    assert response.json()["progress"]["percent"] == 0.0
    response = test_client.post(f"/api/video_tests/{name}/stop")
    assert response.status_code == HTTP_STATUS_OK
    assert response.json()["status"] == "stopped"
    helpers.wait_until_done(runner, name)


def test_start_video_test__bad_request(
    test_client: TestClient, tmp_path: Path, mocker: MockerFixture
) -> None:
    helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
    response = test_client.post(
        "/api/video_tests",
        data={"model": "model_a"},
        files=[("files", ("notes.txt", b"x", "text/plain"))],
    )
    assert response.status_code == HTTP_STATUS_BAD_REQUEST
    response = test_client.get("/api/video_tests/nope")
    assert response.status_code == HTTP_STATUS_BAD_REQUEST
