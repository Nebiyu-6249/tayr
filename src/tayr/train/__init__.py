"""Detector training.

Thin by design. RF-DETR ships the loss, the matcher, the EMA schedule and the PyTorch
Lightning loop; reimplementing any of that would be a second source of truth for
results. What lives here is everything that makes a run *attributable*: a seed, a
resolved device, a manifest written before the first gradient step, and one explicit
mapping from Tayr's config names onto RF-DETR's.
"""

from tayr.train.detector import (
    ABSORBED_KWARGS,
    TrainingRun,
    rfdetr_train_kwargs,
    train_detector,
)

__all__ = [
    "ABSORBED_KWARGS",
    "TrainingRun",
    "rfdetr_train_kwargs",
    "train_detector",
]
