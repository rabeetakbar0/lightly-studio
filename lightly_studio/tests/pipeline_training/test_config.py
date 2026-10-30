import pytest

from lightly_studio.pipeline_training.config import TrainConfig
from lightly_studio.pipeline_training.model_catalog import MODEL_PRESETS


def test_train_config__defaults_match_v4_recipe() -> None:
    config = TrainConfig()
    assert (config.depth, config.width) == MODEL_PRESETS["m"] == (0.67, 0.75)
    assert config.input_size == 640
    assert config.max_epoch == 150
    assert config.warmup_epochs == 5
    assert config.no_aug_epochs == 30
    assert config.basic_lr_per_img == 0.01 / 64.0
    assert config.eval_interval == 1
    assert config.save_history_ckpt is False
    assert config.batch_size == 16
    assert config.fp16 is True
    assert config.baseline == "weights:yolox_m.pth"
    assert config.baseline_parts() == ("weights", "yolox_m.pth")
    assert config.gpu_memory_cap_gb == 15.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_size": 650},
        {"mosaic_scale_min": 3.0, "mosaic_scale_max": 2.0},
        {"baseline": "weights:../x.pth"},
        {"baseline": "yolox_m.pth"},
        {"scheduler": "step"},
    ],
)
def test_train_config__rejects_bad_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"validation error"):
        TrainConfig(**overrides)  # type: ignore[arg-type]


def test_train_config__baseline_parts__none() -> None:
    assert TrainConfig(baseline=None).baseline_parts() is None
    assert TrainConfig(baseline="model:yolox_m_slm_v4").baseline_parts() == (
        "model",
        "yolox_m_slm_v4",
    )
