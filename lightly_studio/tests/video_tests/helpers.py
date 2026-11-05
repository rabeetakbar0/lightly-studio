"""Fixtures for the video test tests: a fake TestApp checkout and a models folder."""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

from pytest_mock import MockerFixture

from lightly_studio.dataset import env
from lightly_studio.video_tests import runner as runner_module
from lightly_studio.video_tests.runner import VideoTestRunner

# The stub test_pipeline.py logs the lines of the real script for every video in
# GATEARM_INPUT_DIR (smallest first) and writes <stem>_processed.mp4 into GATEARM_OUTPUT_DIR.
# A video whose name contains "broken" fails; STUB_SLEEP_SECONDS keeps the process alive.
_PIPELINE_STUB = """
import os, sys, time
input_dir = os.environ["GATEARM_INPUT_DIR"]
output_dir = os.environ["GATEARM_OUTPUT_DIR"]
print("2026-11-02 17:00:00,000 INFO test_pipeline: Model : " + os.environ["GATEARM_MODEL_PATH"])
print("cwd: " + os.getcwd())
time.sleep(float(os.environ.get("STUB_SLEEP_SECONDS", "0")))
videos = sorted(os.listdir(input_dir), key=lambda n: os.path.getsize(os.path.join(input_dir, n)))
for index, name in enumerate(videos, start=1):
    print(f"2026-11-02 17:00:01,000 INFO test_pipeline: [{index}/{len(videos)}] {name} (0.0 MB)")
    if "broken" in name:
        print(f"2026-11-02 17:00:02,000 ERROR test_pipeline: Failed to process {input_dir}/{name}")
        print("Traceback (most recent call last):")
        continue
    print("2026-11-02 17:00:02,000 INFO video_pipeline: Progress 300/600 (50%) | "
          "last-300 avg=10.0ms fps=100.0")
    output = os.path.join(output_dir, os.path.splitext(name)[0] + "_processed.mp4")
    open(output, "wb").write(b"mp4")
    print("2026-11-02 17:00:03,000 INFO test_pipeline: Benchmark stats:")
    print("  fps: 100.0")
    print("2026-11-02 17:00:03,000 INFO test_pipeline: Detection summary:")
    print("  GatearmDetected: True")
    print(f"2026-11-02 17:00:03,000 INFO test_pipeline: Done. Annotated video: {output}")
print("2026-11-02 17:00:04,000 INFO test_pipeline: Batch finished: 1 videos in 3.0 s")
sys.exit(int(os.environ.get("STUB_EXIT_CODE", "0")))
"""


def setup_environment(tmp_path: Path, mocker: MockerFixture) -> VideoTestRunner:
    """Points the TestApp settings at a stub checkout under tmp_path; returns a fresh runner."""
    testapp_dir = tmp_path / "TestApp"
    (testapp_dir / "App").mkdir(parents=True)
    (testapp_dir / "App" / "test_pipeline.py").write_text(textwrap.dedent(_PIPELINE_STUB))
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_a.onnx").write_bytes(b"onnx")
    (models_dir / "model_a_classes.txt").write_text("GateArm\nLedLight\n")

    mocker.patch.object(env, "LIGHTLY_STUDIO_MODELS_DIR", models_dir)
    mocker.patch.object(env, "LIGHTLY_STUDIO_TESTAPP_DIR", testapp_dir)
    mocker.patch.object(env, "LIGHTLY_STUDIO_TESTAPP_PYTHON", sys.executable)
    mocker.patch.object(env, "LIGHTLY_STUDIO_TESTAPP_UPLOADS_DIR", None)
    runner = VideoTestRunner()
    mocker.patch.object(runner_module, "video_test_runner", runner)
    return runner


def wait_until_done(runner: VideoTestRunner, name: str, timeout: float = 20.0) -> None:
    """Waits until the run is no longer running."""
    deadline = time.monotonic() + timeout
    while runner.get_run(name=name).is_active:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Video test {name} did not finish.")
        time.sleep(0.05)
