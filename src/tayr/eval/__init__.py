"""Evaluation harness.

Built before the training pipeline deliberately: the metric definitions are where this
project's research claims live, they are pure logic testable without a GPU, and having
them ready means a training run produces reportable numbers immediately.

`tayr.eval.harness` is deliberately not re-exported here. It reads images and builds
detectors, so importing it needs the `cv` extra; everything named below is pure and must
stay importable without torch or OpenCV.
"""

from tayr.eval.detection import (
    COCO_SWEEP,
    DetectionCounts,
    ImagePrediction,
    MatchResult,
    average_precision,
    evaluate_dataset,
    evaluate_detection,
    match_detections,
    sweep_average_precision,
)
from tayr.eval.false_alarms import FalseAlarmRate, false_alarms_per_hour
from tayr.eval.report import EvaluationReport
from tayr.eval.splits import SplitAssignment, grouped_split

__all__ = [
    "COCO_SWEEP",
    "DetectionCounts",
    "EvaluationReport",
    "FalseAlarmRate",
    "ImagePrediction",
    "MatchResult",
    "SplitAssignment",
    "average_precision",
    "evaluate_dataset",
    "evaluate_detection",
    "false_alarms_per_hour",
    "grouped_split",
    "match_detections",
    "sweep_average_precision",
]
