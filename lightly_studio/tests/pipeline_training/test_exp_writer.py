import ast
from pathlib import Path

from lightly_studio.pipeline_training import exp_writer
from lightly_studio.pipeline_training.config import TrainConfig


def test_render_exp() -> None:
    config = TrainConfig(max_epoch=3, seed=7, gpu_memory_cap_gb=None, mosaic_scale_max=1.5)
    source = exp_writer.render_exp(
        config=config,
        data_dir=Path("/data/slm"),
        num_classes=2,
        output_dir=Path("/runs/r1"),
    )
    ast.parse(source)
    assert "GPU_MEM_CAP_GB = None" in source
    assert "self.exp_name = 'train'" in source
    assert "self.output_dir = '/runs/r1'" in source
    assert "self.data_dir = '/data/slm'" in source
    assert "self.num_classes = 2" in source
    assert "self.depth = 0.67" in source
    assert "self.input_size = (640, 640)" in source
    assert "self.max_epoch = 3" in source
    assert "self.seed = 7" in source
    assert "self.mosaic_scale = (0.1, 1.5)" in source
    assert "self.basic_lr_per_img = 0.00015625" in source
    assert "max_labels=50" in source
