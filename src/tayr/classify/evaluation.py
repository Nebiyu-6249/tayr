"""Hypothesis-test evaluation: per-bucket accuracy with confidence intervals.

The claim under test is that below roughly 20 pixels on target, appearance-based
classification saturates while motion signature stays discriminative. Falsifying it
requires two curves against pixels-on-target, from arms that are otherwise identical.

**Confidence intervals are mandatory, not decorative.** The realistic supply is low
hundreds of tracks (docs/RESEARCH.md 5.3), so a bucket may hold twenty. A point estimate
from twenty tracks carries an interval roughly +/-20 percentage points wide, and quoting
it alone would be an overclaim that the arithmetic here makes impossible to sustain.

Wilson intervals rather than the normal approximation: at small n, or when accuracy is
near 0 or 1, the normal approximation produces bounds outside [0, 1], which is visibly
wrong and quietly wrong just inside those extremes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tayr.geometry import SizeBucket

# 1.96 -> 95%.
_Z = 1.959963984540054


def wilson_interval(successes: int, trials: int, *, z: float = _Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (0.0, 1.0) for zero trials: with no data the true value could be anything,
    and reporting a narrow interval there would be a lie.
    """
    if trials <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > trials:
        raise ValueError(f"successes ({successes}) must be within [0, {trials}]")

    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True, slots=True)
class BucketScore:
    """One arm's performance in one pixels-on-target bucket."""

    bucket: SizeBucket
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.correct, self.total)

    @property
    def is_thin(self) -> bool:
        """Too few tracks for the point estimate to carry meaning on its own."""
        return self.total < 30


@dataclass(slots=True)
class ArmResult:
    """One arm of the comparison, bucketed."""

    name: str
    buckets: dict[SizeBucket, BucketScore] = field(default_factory=dict)

    def accuracy_in(self, bucket: SizeBucket) -> float | None:
        score = self.buckets.get(bucket)
        return score.accuracy if score and score.total else None


@dataclass(slots=True)
class HypothesisResult:
    """The comparison, and what it does or does not license saying."""

    appearance: ArmResult
    motion: ArmResult
    synthetic: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def small_buckets(self) -> tuple[SizeBucket, ...]:
        """Where the hypothesis lives."""
        return (SizeBucket.TINY, SizeBucket.SMALL)

    def motion_wins_in_small_buckets(self) -> bool | None:
        """True / False / None, where None means the data cannot answer.

        Returns None rather than a default when a bucket is empty or the intervals
        overlap: 'we could not tell' is a distinct answer from 'no', and collapsing them
        would turn an under-powered experiment into a negative result.
        """
        decided = False
        for bucket in self.small_buckets:
            appearance = self.appearance.buckets.get(bucket)
            motion = self.motion.buckets.get(bucket)
            if not appearance or not motion or not appearance.total or not motion.total:
                continue
            a_low, a_high = appearance.interval
            m_low, m_high = motion.interval
            if m_low > a_high:
                decided = True
            elif a_low > m_high:
                return False
        return True if decided else None

    def render(self) -> str:
        lines: list[str] = []
        if self.synthetic:
            lines += [
                "!" * 70,
                "!! SYNTHETIC DATA - this is not a result about real drones or birds",
                "!" * 70,
                "",
            ]
        lines += ["Hypothesis test: appearance vs motion, by pixels-on-target", "=" * 70, ""]
        header = f"  {'bucket':10s}  {'appearance':>26s}  {'motion':>26s}"
        lines += [header, "  " + "-" * (len(header) - 2)]

        for bucket in (SizeBucket.TINY, SizeBucket.SMALL, SizeBucket.MEDIUM, SizeBucket.LARGE):
            cells = []
            for arm in (self.appearance, self.motion):
                score = arm.buckets.get(bucket)
                if not score or not score.total:
                    cells.append(f"{'no tracks':>26s}")
                    continue
                low, high = score.interval
                flag = "*" if score.is_thin else " "
                cells.append(
                    f"{score.accuracy:.3f} [{low:.2f}-{high:.2f}] n={score.total:<4d}{flag}"
                )
            lines.append(f"  {bucket.value:10s}  {cells[0]}  {cells[1]}")

        lines += [
            "",
            "  * fewer than 30 tracks: the interval, not the point estimate, is the result",
        ]

        verdict = self.motion_wins_in_small_buckets()
        lines += ["", "  Verdict:"]
        if verdict is None:
            lines.append(
                "    UNDETERMINED. The intervals overlap or a bucket is empty, so this "
                "data cannot\n    answer the question. That is not evidence against the "
                "hypothesis - it is an\n    under-powered experiment, and the writeup must say so."
            )
        elif verdict:
            lines.append(
                "    Motion beat appearance in at least one small bucket with "
                "non-overlapping\n    intervals."
            )
        else:
            lines.append(
                "    Appearance beat motion in a small bucket with non-overlapping "
                "intervals.\n    THE HYPOTHESIS IS WRONG and the writeup says so."
            )

        if self.notes:
            lines += ["", "  NOTES:"] + [f"    - {n}" for n in self.notes]
        return "\n".join(lines)


def compare_arms(
    appearance: ArmResult, motion: ArmResult, *, synthetic: bool = False
) -> HypothesisResult:
    """Assemble the comparison, warning about anything that weakens it."""
    notes: list[str] = []
    for bucket in (SizeBucket.TINY, SizeBucket.SMALL, SizeBucket.MEDIUM, SizeBucket.LARGE):
        a = appearance.buckets.get(bucket)
        m = motion.buckets.get(bucket)
        if a and m and a.total != m.total:
            notes.append(
                f"{bucket.value}: arms saw different track counts ({a.total} vs {m.total}). "
                "The comparison is only controlled if both arms classify the SAME tracks."
            )
    return HypothesisResult(appearance=appearance, motion=motion, synthetic=synthetic, notes=notes)
