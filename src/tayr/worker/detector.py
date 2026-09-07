"""Detector interface.

The detector sits behind a Protocol so RF-DETR, D-FINE, or anything else can be swapped
without touching the pipeline. Phase 0 decision D2 selected RF-DETR (Apache-2.0);
Ultralytics is deliberately absent because it is AGPL-3.0 and would relicense the whole
project.

`StubDetector` carries `stub_` in its name on purpose. Per this project's failure
policy, a stand-in for unimplemented functionality must be unmistakable, and any run
that uses one is marked `synthetic=True` so the label reaches the API response, the UI,
and any generated report. It exists so the pipeline can be tested end to end on a
machine with no GPU; it is not a detector and produces no real detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from tayr.geometry import FloatArray


@dataclass(frozen=True, slots=True)
class Detection:
    """Detections in one frame: xyxy boxes with confidences."""

    boxes_xyxy: FloatArray
    scores: npt.NDArray[np.float64]
    class_ids: npt.NDArray[np.int64] | None = None
    """Predicted class per box, where the backend produces one.

    Optional because the tracker and every size-bucketed detection metric are
    class-agnostic - they ask where things are, not what they are. It is carried anyway
    because a multi-class detector emits it and an adapter that silently dropped it
    would make the classifier comparison impossible to wire up later."""

    def __post_init__(self) -> None:
        if len(self.boxes_xyxy) != len(self.scores):
            raise ValueError(f"{len(self.boxes_xyxy)} box(es) but {len(self.scores)} score(s)")
        if self.class_ids is not None and len(self.class_ids) != len(self.boxes_xyxy):
            raise ValueError(
                f"{len(self.boxes_xyxy)} box(es) but {len(self.class_ids)} class id(s)"
            )


@runtime_checkable
class Detector(Protocol):
    """What the pipeline requires of a detector."""

    @property
    def is_real(self) -> bool:
        """False for any stand-in. A run using one is recorded as synthetic.

        This is a property on the interface rather than an isinstance check so that a
        future stub cannot be introduced without answering the question.
        """
        ...

    @property
    def name(self) -> str:
        """Recorded in the run manifest."""
        ...

    def detect(self, frame: npt.NDArray[np.uint8]) -> Detection:
        """Detect in one HxWx3 RGB frame."""
        ...


class StubDetector:
    """A stand-in that detects nothing. NOT A DETECTOR.

    Returns zero detections for every frame, which is honest: it has no model, so it has
    found nothing. It does not invent plausible boxes - fabricated detections would flow
    into tracks, features, and a report, and would be indistinguishable from real ones
    by the time anyone looked.
    """

    @property
    def is_real(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "stub_detector"

    def detect(self, frame: npt.NDArray[np.uint8]) -> Detection:
        del frame
        return Detection(
            boxes_xyxy=np.empty((0, 4), dtype=np.float64),
            scores=np.empty(0, dtype=np.float64),
        )


class ScriptedDetector:
    """A test double that replays a fixed sequence of detections. NOT A DETECTOR.

    Lets the pipeline's tracking and feature stages be exercised against known input, so
    a pipeline bug is distinguishable from a detector bug. Also marks its runs synthetic.
    """

    def __init__(self, per_frame: list[Detection]) -> None:
        self._per_frame = per_frame

    @property
    def is_real(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "stub_scripted_detector"

    def detect(self, frame: npt.NDArray[np.uint8]) -> Detection:
        del frame
        if not self._per_frame:
            return Detection(np.empty((0, 4), dtype=np.float64), np.empty(0, dtype=np.float64))
        return self._per_frame.pop(0)
