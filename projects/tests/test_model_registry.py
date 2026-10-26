from pathlib import Path

from pipeline import model_registry


def test_discover_models(tmp_path: Path) -> None:
    (tmp_path / "b_model.onnx").write_bytes(b"")
    (tmp_path / "b_model_classes.txt").write_text("GateArm\n\nLedLight\n")
    (tmp_path / "a_model.onnx").write_bytes(b"")
    (tmp_path / "a_model_classes.txt").write_text("person\n")
    (tmp_path / "no_classes.onnx").write_bytes(b"")
    (tmp_path / "empty.onnx").write_bytes(b"")
    (tmp_path / "empty_classes.txt").write_text("\n")
    (tmp_path / "checkpoint.pth").write_bytes(b"")

    models = model_registry.discover_models(models_dir=tmp_path)

    assert [m.name for m in models] == ["a_model", "b_model"]
    assert models[1].class_names == ("GateArm", "LedLight")
    assert models[1].onnx_path == tmp_path / "b_model.onnx"
    assert models[1].default_annotation_source == "b_model_prediction"


def test_discover_models__missing_dir(tmp_path: Path) -> None:
    assert model_registry.discover_models(models_dir=tmp_path / "nope") == []
