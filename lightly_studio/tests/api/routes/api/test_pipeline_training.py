from pathlib import Path

from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from lightly_studio.api.routes.api.status import (
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_CREATED,
    HTTP_STATUS_OK,
    HTTP_STATUS_UNPROCESSABLE_ENTITY,
)
from tests.pipeline_training import helpers


def test_get_training_options(
    test_client: TestClient, tmp_path: Path, mocker: MockerFixture
) -> None:
    helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
    helpers.write_model(tmp_path, "yolox_m_slm_v4", ["GateArm", "LedLight"])
    body = test_client.get("/api/pipeline_training/options").json()
    assert body["defaults"]["max_epoch"] == 150
    assert body["defaults"]["baseline"] == "weights:yolox_m.pth"
    assert body["presets"]["m"] == [0.67, 0.75]
    assert body["baselines"] == [
        {
            "value": "model:yolox_m_slm_v4",
            "label": "yolox_m_slm_v4",
            "group": "Trained models",
            "depth": 0.67,
            "width": 0.75,
            "classes": ["GateArm", "LedLight"],
            "available": False,
        },
        {
            "value": "weights:yolox_m.pth",
            "label": "yolox_m.pth",
            "group": "Stock YOLOX",
            "depth": 0.67,
            "width": 0.75,
            "classes": [],
            "available": True,
        },
    ]
    assert body["environment"]["ready"] is True
    assert body["environment"]["weights"] == ["yolox_m.pth"]


def test_training_runs_flow(test_client: TestClient, tmp_path: Path, mocker: MockerFixture) -> None:
    runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
    helpers.write_dataset(tmp_path, "data1", ["GateArm", "LedLight"])
    assert test_client.get("/api/pipeline_training/runs").json() == []

    response = test_client.post(
        "/api/pipeline_training/runs",
        json={"name": " run1 ", "dataset": "data1", "config": {"max_epoch": 2, "batch_size": 8}},
    )
    assert response.status_code == HTTP_STATUS_CREATED, response.text
    assert response.json()["name"] == "run1"
    assert response.json()["config"]["batch_size"] == 8
    assert response.json()["config"]["warmup_epochs"] == 5
    helpers.wait_until_done(runner, "run1")

    body = test_client.get("/api/pipeline_training/runs/run1").json()
    assert body["status"] == "finished"
    assert body["classes"] == ["GateArm", "LedLight"]
    assert body["progress"]["epoch"] == 1
    assert body["progress"]["max_epoch"] == 2
    assert body["progress"]["best_ap50_95"] == 0.5
    assert body["progress"]["evaluations"] == [{"epoch": 1, "ap50": 0.8, "ap50_95": 0.5}]
    assert body["onnx_path"] == str(tmp_path / "models" / "run1.onnx")

    models = test_client.get("/api/pipeline_models").json()
    assert [m["name"] for m in models] == ["run1"]
    assert models[0]["has_checkpoint"] is True
    assert models[0]["card"]["dataset"]["name"] == "data1"
    assert models[0]["card"]["dataset"]["images"] == 2
    assert models[0]["card"]["config"]["batch_size"] == 8
    assert models[0]["card"]["best_ap50_95"] == 0.5
    assert models[0]["card"]["evaluations"] == [{"epoch": 1, "ap50": 0.8, "ap50_95": 0.5}]

    log = test_client.get("/api/pipeline_training/runs/run1/log", params={"lines": 3}).json()
    assert len(log["text"].splitlines()) == 3

    assert [r["name"] for r in test_client.get("/api/pipeline_training/runs").json()] == ["run1"]
    response = test_client.post("/api/pipeline_training/runs/run1/stop")
    assert response.status_code == HTTP_STATUS_BAD_REQUEST

    response = test_client.post(
        "/api/pipeline_training/runs",
        json={"name": "bad", "dataset": "data1", "config": {"input_size": 650}},
    )
    assert response.status_code == HTTP_STATUS_UNPROCESSABLE_ENTITY, response.text
    response = test_client.post(
        "/api/pipeline_training/runs",
        json={"name": "bad", "dataset": "missing", "config": {}},
    )
    assert response.status_code == HTTP_STATUS_BAD_REQUEST, response.text

    assert test_client.delete("/api/pipeline_training/runs/run1").status_code == HTTP_STATUS_OK
    assert (
        test_client.get("/api/pipeline_training/runs/run1").status_code == HTTP_STATUS_BAD_REQUEST
    )


def test_test_runs_flow(test_client: TestClient, tmp_path: Path, mocker: MockerFixture) -> None:
    runner = helpers.setup_environment(tmp_path=tmp_path, mocker=mocker)
    helpers.write_dataset(tmp_path, "data1", ["GateArm"])
    helpers.write_dataset(tmp_path, "test1", ["GateArm"], splits={"test2017": ["s1", "s2"]})
    helpers.write_model(tmp_path, "m1", ["GateArm"], checkpoint=True)

    assert test_client.get("/api/pipeline_training/test_datasets").json() == [
        {"name": "test1", "classes": ["GateArm"], "images": 2, "kind": "train"}
    ]

    response = test_client.post(
        "/api/pipeline_training/tests", json={"model": "m1", "dataset": "test1", "batch_size": 2}
    )
    assert response.status_code == HTTP_STATUS_CREATED, response.text
    name = response.json()["name"]
    assert response.json()["kind"] == "test"
    assert response.json()["model"] == "m1"
    helpers.wait_until_done(runner, name)

    body = test_client.get(f"/api/pipeline_training/runs/{name}").json()
    assert body["status"] == "finished"
    assert body["test_result"]["ap50_95"] == 0.763
    assert body["test_result"]["per_class_ap"] == {"GateArm": 0.82505, "LedLight": 0.70168}
    assert body["test_result"]["overlap_images"] is None

    models = test_client.get("/api/pipeline_models").json()
    assert models[0]["card"]["trained"] is False
    assert models[0]["card"]["tests"][0]["dataset"] == "test1"

    response = test_client.post(
        "/api/pipeline_training/tests", json={"model": "m1", "dataset": "data1"}
    )
    assert response.status_code == HTTP_STATUS_BAD_REQUEST
