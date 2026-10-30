from lightly_studio.pipeline_training import log_metrics

LOG = """
10:00:00 | INFO | ---> start train epoch1
10:00:05 | INFO | epoch: 1/150, iter: 20/557, total_loss: 9.1, lr: 1.0e-05, size: 640, ETA: 9:30:00
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.456
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.100
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
10:04:01 | INFO | ---> start train epoch2
10:04:20 | INFO | epoch: 2/150, iter: 40/557, total_loss: 8.1, lr: 2.0e-05, size: 672, ETA: 9:20:00
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.234
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.567
"""


def test_parse_log() -> None:
    progress = log_metrics.parse_log(text=LOG)
    assert (progress.epoch, progress.max_epoch) == (2, 150)
    assert (progress.iteration, progress.max_iteration) == (40, 557)
    assert progress.eta == "9:20:00"
    assert [(e.epoch, e.ap50_95, e.ap50) for e in progress.evaluations] == [
        (1, 0.123, 0.456),
        (2, 0.234, 0.567),
    ]
    assert progress.best_ap50_95 == 0.234
    assert progress.last_line.endswith("= 0.567")


def test_parse_log__empty() -> None:
    progress = log_metrics.parse_log(text="")
    assert progress.epoch == 0
    assert progress.evaluations == []
    assert progress.best_ap50_95 is None
