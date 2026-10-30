"""Initialize environment variables for the dataset module."""

from pathlib import Path
from typing import Optional

from environs import Env

env = Env()
env.read_env()
LIGHTLY_STUDIO_EMBEDDINGS_MODEL_TYPE: str = env.str(
    "LIGHTLY_STUDIO_EMBEDDINGS_MODEL_TYPE", "MOBILE_CLIP"
)
LIGHTLY_STUDIO_MODEL_CACHE_DIR: Path = env.path(
    "LIGHTLY_STUDIO_MODEL_CACHE_DIR", Path.home() / ".cache" / "lightly-studio"
)
LIGHTLY_STUDIO_PROTOCOL: str = env.str("LIGHTLY_STUDIO_PROTOCOL", "http")
# Files uploaded through the GUI are stored here, in one subfolder per collection.
LIGHTLY_STUDIO_UPLOAD_DIR: Path = env.path(
    "LIGHTLY_STUDIO_UPLOAD_DIR", Path.cwd() / "lightly_studio_uploads"
)
# COCO datasets prepared from the GUI are written here, one folder per dataset.
LIGHTLY_STUDIO_DATASETS_DIR: Path = env.path(
    "LIGHTLY_STUDIO_DATASETS_DIR", Path.cwd() / "lightly_studio_datasets"
)
# Training runs started from the GUI, one folder per run.
LIGHTLY_STUDIO_RUNS_DIR: Path = env.path(
    "LIGHTLY_STUDIO_RUNS_DIR", Path.cwd() / "lightly_studio_runs"
)
# Exported ONNX models land here, next to their class lists.
LIGHTLY_STUDIO_MODELS_DIR: Path = env.path(
    "LIGHTLY_STUDIO_MODELS_DIR", Path.cwd() / "lightly_studio_models"
)
# Pretrained YOLOX checkpoints (.pth) that a run can start from.
LIGHTLY_STUDIO_WEIGHTS_DIR: Path = env.path(
    "LIGHTLY_STUDIO_WEIGHTS_DIR", Path.cwd() / "lightly_studio_weights"
)
# The YOLOX checkout with tools/train.py, and the interpreter that has YOLOX and torch
# installed. Training is off until both are set.
LIGHTLY_STUDIO_YOLOX_DIR: Optional[Path] = env.path("LIGHTLY_STUDIO_YOLOX_DIR", default=None)
LIGHTLY_STUDIO_TRAIN_PYTHON: Optional[str] = env.str("LIGHTLY_STUDIO_TRAIN_PYTHON", default=None)
LIGHTLY_STUDIO_PORT: int = env.int("LIGHTLY_STUDIO_PORT", 8001)
LIGHTLY_STUDIO_HOST: str = env.str("LIGHTLY_STUDIO_HOST", "localhost")
LIGHTLY_STUDIO_DEBUG: bool = env.bool("LIGHTLY_STUDIO_DEBUG", False)

LIGHTLY_STUDIO_DATABASE_URL: Optional[str] = env.str("LIGHTLY_STUDIO_DATABASE_URL", default=None)

LIGHTLY_STUDIO_API_URL: Optional[str] = env.str("LIGHTLY_STUDIO_API_URL", default=None)
LIGHTLY_STUDIO_TOKEN: Optional[str] = env.str("LIGHTLY_STUDIO_TOKEN", default=None)
LIGHTLY_STUDIO_API_KEY: Optional[str] = env.str("LIGHTLY_STUDIO_API_KEY", default=None)

LIGHTLY_STUDIO_REMOTE_IMAGE_PROBE_WORKERS: int = max(
    1, env.int("LIGHTLY_STUDIO_REMOTE_IMAGE_PROBE_WORKERS", 32)
)

# The address policy of a remote embedding server, see embed/remote/url_policy.py.
# True keeps every address usable. False requires https and a public address.
LIGHTLY_STUDIO_REMOTE_EMBEDDER_ALLOW_PRIVATE_URLS: bool = env.bool(
    "LIGHTLY_STUDIO_REMOTE_EMBEDDER_ALLOW_PRIVATE_URLS", True
)

LIGHTLY_STUDIO_REQUEST_TIMING_ENABLED: bool = env.bool(
    "LIGHTLY_STUDIO_REQUEST_TIMING_ENABLED", False
)
LIGHTLY_STUDIO_REQUEST_TIMING_ERROR_MS: int = env.int("LIGHTLY_STUDIO_REQUEST_TIMING_ERROR_MS", 200)
LIGHTLY_STUDIO_REQUEST_TIMING_FAIL_ON_ERROR: bool = env.bool(
    "LIGHTLY_STUDIO_REQUEST_TIMING_FAIL_ON_ERROR", False
)
