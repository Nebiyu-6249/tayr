"""Evaluation harness.

Built before the training pipeline deliberately: the metric definitions are where this
project's research claims live, they are pure logic testable without a GPU, and having
them ready means a training run produces reportable numbers immediately.
"""

from tayr.eval.detection import (
    DetectionCounts,
    MatchResult,
    average_precision,
    evaluate_detection,
    match_detections,
)
from tayr.eval.false_alarms import FalseAlarmRate, false_alarms_per_hour
from tayr.eval.report import EvaluationReport
from tayr.eval.splits import SplitAssignment, grouped_split

__all__ = [
    "DetectionCounts",
    "EvaluationReport",
    "FalseAlarmRate",
    "MatchResult",
    "SplitAssignment",
    "average_precision",
    "evaluate_detection",
    "false_alarms_per_hour",
    "grouped_split",
    "match_detections",
]
