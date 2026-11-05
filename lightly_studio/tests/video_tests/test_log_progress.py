from lightly_studio.video_tests import log_progress

_LOG = """\
2026-11-02 17:00:00,000 INFO test_pipeline: Input : /up/x (3 videos, smallest first)
2026-11-02 17:00:01,000 INFO test_pipeline: [1/3] a.mp4 (1.2 MB)
2026-11-02 17:00:02,000 INFO video_pipeline: Progress 300/300 (100%) | last-300 avg=9.0ms fps=111.1
2026-11-02 17:00:02,000 INFO video_pipeline: All frames processed
2026-11-02 17:00:03,000 INFO test_pipeline: Benchmark stats:
  frames_processed: 300
  fps: 111.1
2026-11-02 17:00:03,000 INFO test_pipeline: Wall-clock time: 3.10 s
2026-11-02 17:00:03,000 INFO test_pipeline: Detection summary:
  GatearmDetected: True
  TipLightBlinkCount: 12
2026-11-02 17:00:03,000 INFO test_pipeline: Done. Annotated video: /res/a_processed.mp4
2026-11-02 17:00:04,000 INFO test_pipeline: [2/3] broken.mp4 (2.0 MB)
2026-11-02 17:00:04,000 ERROR test_pipeline: Failed to process /up/x/broken.mp4
Traceback (most recent call last):
  File "x.py", line 1
2026-11-02 17:00:05,000 INFO test_pipeline: [3/3] c.mp4 (5.0 MB)
2026-11-02 17:00:06,000 INFO video_pipeline: Progress 300/1200 (25%) | last-300 avg=10.0ms fps=100.0
2026-11-02 17:00:07,000 WARNING video_pipeline: Primary LED locations did not give the best \
results, re-running the pipeline with backup locations.
2026-11-02 17:00:08,000 INFO video_pipeline: Progress 600/1200 (50%) | last-300 avg=12.5ms fps=80.0
"""


def test_parse_progress__running() -> None:
    progress = log_progress.parse_progress(text=_LOG)
    assert progress.total == 3
    assert progress.current_index == 3
    assert progress.current_video == "c.mp4"
    assert (progress.frames_done, progress.frames_total) == (600, 1200)
    assert progress.fps == 80.0
    assert progress.pass_number == 2
    assert not progress.batch_finished
    # Two videos ended, the third is half way.
    assert progress.percent == 83.3
    a, broken, c = progress.videos
    assert a.status == log_progress.VIDEO_DONE
    assert a.output_path == "/res/a_processed.mp4"
    assert a.summary == {"GatearmDetected": "True", "TipLightBlinkCount": "12"}
    assert a.benchmark == {"frames_processed": "300", "fps": "111.1"}
    assert broken.status == log_progress.VIDEO_FAILED
    assert broken.summary == {}
    assert c.status == log_progress.VIDEO_PROCESSING


def test_parse_progress__batch_finished() -> None:
    progress = log_progress.parse_progress(
        text=_LOG + "2026-11-02 17:00:09,000 INFO test_pipeline: Batch finished: 3 videos\n"
    )
    assert progress.batch_finished
    assert progress.percent == 100.0


def test_parse_progress__empty() -> None:
    progress = log_progress.parse_progress(text="")
    assert progress.total == 0
    assert progress.videos == []
    assert progress.percent == 0.0
