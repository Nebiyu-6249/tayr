"""Track classification: the two arms of the hypothesis test.

Phase 0 decision D4 requires a *controlled* comparison. Comparing a detector's mAP
against a classifier's accuracy is not one - those are different quantities with
different units. So both arms classify the same tracks, in the same pixel buckets, with
the same metric, on the same grouped splits.
"""

from tayr.classify.evaluation import (
    ArmResult,
    HypothesisResult,
    compare_arms,
    wilson_interval,
)
from tayr.classify.motion import MotionClassifier

__all__ = [
    "ArmResult",
    "HypothesisResult",
    "MotionClassifier",
    "compare_arms",
    "wilson_interval",
]
