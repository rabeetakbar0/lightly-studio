from pathlib import Path

from lightly_studio.pipeline_training import model_catalog
from lightly_studio.pipeline_training.model_catalog import ModelCard
from tests.pipeline_training import helpers


def test_list_models(tmp_path: Path) -> None:
    models_dir = helpers.write_model(tmp_path, "yolox_m_slm_v4", ["GateArm", "LedLight"])
    helpers.write_model(tmp_path, "run1", ["GateArm"], checkpoint=True)
    (models_dir / "orphan.onnx").write_bytes(b"x")  # No class list: skipped.
    card = ModelCard(
        run_name="run1",
        dataset={"name": "data1", "classes": ["GateArm"]},
        baseline="weights:yolox_m.pth",
        config={"depth": 0.33, "width": 0.5, "max_epoch": 3},
        best_ap50_95=0.5,
    )
    model_catalog.write_card(models_dir=models_dir, name="run1", card=card)

    models = {m.name: m for m in model_catalog.list_models(models_dir=models_dir)}
    assert set(models) == {"yolox_m_slm_v4", "run1"}
    v4 = models["yolox_m_slm_v4"]
    assert v4.classes == ["GateArm", "LedLight"]
    assert v4.checkpoint_path is None
    assert v4.card is None
    assert (v4.depth, v4.width) == (0.67, 0.75)  # Guessed from the name.
    run1 = models["run1"]
    assert run1.checkpoint_path == models_dir / "run1.pth"
    assert run1.card == card
    assert (run1.depth, run1.width) == (0.33, 0.5)  # From the card.
    assert run1.size_bytes == 4


def test_list_models__missing_folder(tmp_path: Path) -> None:
    assert model_catalog.list_models(models_dir=tmp_path / "none") == []


def test_size_from_name() -> None:
    assert model_catalog.size_from_name(name="yolox_m.pth") == (0.67, 0.75)
    assert model_catalog.size_from_name(name="yolox_l_slm") == (1.0, 1.0)
    assert model_catalog.size_from_name(name="yolox_mini") is None
    assert model_catalog.size_from_name(name="helper_model_v1") is None
