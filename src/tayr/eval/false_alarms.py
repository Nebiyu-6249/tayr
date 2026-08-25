"""False alarms per hour on drone-free footage.

This is the number that decides whether a system like Tayr is usable, and it is
under-reported in the literature. It needs no annotation: on footage guaranteed to
contain no drone, **every** detection is a false positive. That makes it cheap to
measure and impossible to game by tuning a threshold on the test set, because raising
the threshold to cut false alarms visibly costs recall in the bucketed mAP alongside it.

The pairing matters. False-alarm rate alone is trivially optimised by detecting nothing;
mAP alone is trivially optimised by detecting everything. Report both or neither.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tayr.errors import ConfigError


@dataclass(frozen=True, slots=True)
class FalseAlarmRate:
    """Measured on footage asserted to contain no target of interest."""

    n_false_alarms: int
    duration_hours: float
    n_frames: int
    confidence_threshold: float

    @property
    def per_hour(self) -> float:
        return self.n_false_alarms / self.duration_hours

    @property
    def per_frame(self) -> float:
        return self.n_false_alarms / self.n_frames if self.n_frames else 0.0

    def render(self) -> str:
        return (
            f"false alarms: {self.n_false_alarms} over {self.duration_hours:.3f} h "
            f"({self.n_frames} frames) at conf>={self.confidence_threshold:.2f}\n"
            f"  = {self.per_hour:.1f} / hour  ({self.per_frame:.4f} / frame)"
        )


def false_alarms_per_hour(
    pred_scores_per_frame: list[npt.NDArray[np.float64]],
    *,
    fps: float,
    confidence_threshold: float,
) -> FalseAlarmRate:
    """Count detections on negative footage.

    `pred_scores_per_frame` has one entry per frame - an array of detection confidences,
    empty where the detector fired nothing. Frames with no detections must still be
    present: they are the denominator, and omitting them inflates the per-frame rate.

    Raises rather than assuming a frame rate, because a wrong fps scales the headline
    number linearly and silently.
    """
    if fps <= 0:
        raise ConfigError(f"fps must be positive; got {fps}. Probe it from the container.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ConfigError(f"confidence_threshold must be in [0, 1]; got {confidence_threshold}")
    if not pred_scores_per_frame:
        raise ConfigError("no frames supplied. A false-alarm rate over zero frames is not a rate.")

    n_frames = len(pred_scores_per_frame)
    n_alarms = sum(
        int(np.sum(np.asarray(scores, dtype=np.float64) >= confidence_threshold))
        for scores in pred_scores_per_frame
    )

    return FalseAlarmRate(
        n_false_alarms=n_alarms,
        duration_hours=n_frames / fps / 3600.0,
        n_frames=n_frames,
        confidence_threshold=confidence_threshold,
    )
