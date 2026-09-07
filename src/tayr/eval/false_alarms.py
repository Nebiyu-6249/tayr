"""False alarms per hour on drone-free footage.

This is the number that decides whether a system like Tayr is usable, and it is
under-reported in the literature. It needs no annotation: on footage guaranteed to
contain no drone, **every** detection is a false positive. That makes it cheap to
measure and impossible to game by tuning a threshold on the test set, because raising
the threshold to cut false alarms visibly costs recall in the bucketed mAP alongside it.

The pairing matters. False-alarm rate alone is trivially optimised by detecting nothing;
mAP alone is trivially optimised by detecting everything. Report both or neither.

**A rate needs an interval.** Zero false alarms over one second of footage is "0.0 per
hour" and means nothing; zero over four hours means something. Both render identically
as a point estimate, so this module reports an exact Poisson interval next to the rate.
With `n` alarms in `T` hours the 95% upper bound is roughly `3/T` when `n` is zero, which
is what stops a short negative clip being quoted as a clean sheet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from tayr.errors import ConfigError

#: Two-sided confidence level for the reported interval.
CONFIDENCE_LEVEL = 0.95


#: Below this much negative footage, the per-hour figure is reported with a warning.
#: Half an hour of negatives puts the zero-alarm 95% upper bound at about 6/hour, which
#: is the point at which the number starts being able to rule anything out.
MIN_CREDIBLE_HOURS = 0.5


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

    @property
    def interval_95_per_hour(self) -> tuple[float, float]:
        """Exact (Garwood) Poisson interval on the rate, in alarms per hour.

        The count is Poisson in the observation window, so the interval comes from the
        chi-square quantiles rather than a normal approximation - which is the right
        choice precisely in the case that matters here, where the count is 0 or 1 and a
        normal approximation would give a nonsense interval of zero width.
        """
        alpha = 1.0 - CONFIDENCE_LEVEL
        n = self.n_false_alarms
        lower = 0.0 if n == 0 else float(chi2.ppf(alpha / 2, 2 * n) / 2)
        upper = float(chi2.ppf(1 - alpha / 2, 2 * (n + 1)) / 2)
        return lower / self.duration_hours, upper / self.duration_hours

    def render(self) -> str:
        low, high = self.interval_95_per_hour
        lines = [
            f"false alarms: {self.n_false_alarms} over {self.duration_hours:.4f} h "
            f"({self.n_frames} frames) at conf>={self.confidence_threshold:.2f}",
            f"  = {self.per_hour:.1f} / hour   95% CI [{low:.1f}, {high:.1f}]"
            f"   ({self.per_frame:.4f} / frame)",
        ]
        if self.duration_hours < MIN_CREDIBLE_HOURS:
            lines.append(
                f"  NOT ENOUGH FOOTAGE. {self.duration_hours * 3600:.0f}s of negatives "
                f"puts the upper bound at {high:.0f}/hour, which rules out nothing. Quote "
                f"this only alongside the interval, and get at least "
                f"{MIN_CREDIBLE_HOURS:.1f} h of drone-free footage before using it as a "
                "headline number."
            )
        return "\n".join(lines)


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
