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
from tayr.geometry import SizeBucket, pixels_on_target, size_bucket

#: Two-sided confidence level for the reported interval.
CONFIDENCE_LEVEL = 0.95


#: Below this much negative footage, the per-hour figure is reported with a warning.
#: Half an hour of negatives puts the zero-alarm 95% upper bound at about 6/hour, which
#: is the point at which the number starts being able to rule anything out.
MIN_CREDIBLE_HOURS = 0.5


#: Confidence bins for the histogram. A false-alarm rate at the operating point says how
#: many; the distribution says how BADLY - a detector firing on birds at 0.9 is a
#: different problem from one firing at 0.3, and only the second can be thresholded away.
_CONFIDENCE_BINS: tuple[float, ...] = (0.25, 0.5, 0.7, 0.8, 0.9, 1.01)


@dataclass(frozen=True, slots=True)
class FalseAlarmProfile:
    """What the false alarms looked like, not just how many there were.

    A per-hour rate answers "how noisy". It cannot answer "can I threshold this away" or
    "which targets does it fire on", and both change what should be done about it. Kept
    separate from the rate so that a caller with no boxes still gets a rate.
    """

    scores: tuple[float, ...]
    pixels_on_target: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.scores) != len(self.pixels_on_target):
            raise ConfigError(
                f"{len(self.scores)} score(s) against {len(self.pixels_on_target)} "
                "size(s): every false alarm needs both, or the buckets do not add up."
            )

    @property
    def n(self) -> int:
        return len(self.scores)

    def by_bucket(self) -> dict[str, int]:
        """False alarms per pixels-on-target bucket, every bucket present."""
        counts = dict.fromkeys((b.value for b in SizeBucket), 0)
        for pot in self.pixels_on_target:
            # size_bucket takes boxes; build a square of the right side length so one
            # implementation of the edges is used rather than two.
            side = float(pot)
            bucket = str(size_bucket(np.array([[0.0, 0.0, side, side]]))[0])
            counts[bucket] += 1
        return counts

    def confidence_histogram(self) -> dict[str, int]:
        """Counts by confidence band, so the distribution is visible, not just the mean."""
        bins: dict[str, int] = {}
        low = 0.0
        for high in _CONFIDENCE_BINS:
            label = f"{low:.2f}-{min(high, 1.0):.2f}"
            bins[label] = sum(1 for s in self.scores if low <= s < high)
            low = high
        return bins

    def render(self) -> str:
        if not self.scores:
            return "  no false alarms to characterise."
        arr = np.asarray(self.scores, dtype=np.float64)
        lines = [
            f"  confidence of the {self.n} false alarm(s): "
            f"median {float(np.median(arr)):.2f}, max {float(arr.max()):.2f}",
        ]
        lines += [f"    {band:12s} {n:>6d}" for band, n in self.confidence_histogram().items()]
        lines.append("  by pixels-on-target:")
        lines += [f"    {bucket:12s} {n:>6d}" for bucket, n in self.by_bucket().items()]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FalseAlarmRate:
    """Measured on footage asserted to contain no target of interest."""

    n_false_alarms: int
    duration_hours: float
    n_frames: int
    confidence_threshold: float
    profile: FalseAlarmProfile | None = None
    """Present when the caller supplied boxes as well as scores."""

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
        if self.profile is not None and self.profile.n:
            lines.append(self.profile.render())
            if self.profile.by_bucket()[SizeBucket.LARGE.value] >= max(1, self.profile.n // 2):
                # Worth saying out loud because it inverts the usual expectation: small
                # targets are assumed to be the hard case, and a detector that fires
                # hardest on large close objects is failing at discrimination rather
                # than at sensitivity - which no threshold fixes.
                lines.append(
                    "  MOST FALSE ALARMS ARE LARGE TARGETS. This is discrimination "
                    "failing, not sensitivity: the detector is confidently boxing big, "
                    "close, high-contrast objects that are not drones. Raising the "
                    "confidence threshold would cost real detections before it removed "
                    "these."
                )
        return "\n".join(lines)


def false_alarms_per_hour(
    pred_scores_per_frame: list[npt.NDArray[np.float64]],
    *,
    fps: float,
    confidence_threshold: float,
    pred_boxes_per_frame: list[npt.NDArray[np.float64]] | None = None,
) -> FalseAlarmRate:
    """Count detections on negative footage.

    `pred_scores_per_frame` has one entry per frame - an array of detection confidences,
    empty where the detector fired nothing. Frames with no detections must still be
    present: they are the denominator, and omitting them inflates the per-frame rate.

    `pred_boxes_per_frame`, when supplied, adds a `FalseAlarmProfile`: the confidence
    distribution and the pixels-on-target breakdown. Optional because a rate is still a
    rate without it, and a caller that has only scores should not be forced to invent
    geometry.

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

    profile: FalseAlarmProfile | None = None
    if pred_boxes_per_frame is not None:
        if len(pred_boxes_per_frame) != n_frames:
            raise ConfigError(
                f"{len(pred_boxes_per_frame)} frame(s) of boxes against {n_frames} of "
                "scores. They index the same frames; a mismatch means one of them is "
                "missing the empty frames that form the denominator."
            )
        kept_scores: list[float] = []
        kept_sizes: list[float] = []
        for scores, boxes in zip(pred_scores_per_frame, pred_boxes_per_frame, strict=True):
            arr = np.asarray(scores, dtype=np.float64).reshape(-1)
            box_arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
            if len(box_arr) != len(arr):
                raise ConfigError(
                    f"{len(box_arr)} box(es) against {len(arr)} score(s) in one frame."
                )
            above = arr >= confidence_threshold
            if not above.any():
                continue
            kept_scores.extend(float(v) for v in arr[above])
            kept_sizes.extend(float(v) for v in pixels_on_target(box_arr[above]))
        profile = FalseAlarmProfile(scores=tuple(kept_scores), pixels_on_target=tuple(kept_sizes))

    return FalseAlarmRate(
        n_false_alarms=n_alarms,
        duration_hours=n_frames / fps / 3600.0,
        n_frames=n_frames,
        confidence_threshold=confidence_threshold,
        profile=profile,
    )
