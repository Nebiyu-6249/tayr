"""Fit what can be fitted, and describe what cannot.

At the sample size this dataset starts from - ten tracks across three classes, from eight
clips - there is no inferential result available and pretending otherwise would be the
worst thing this module could do. So it does two separate things and labels which is
which:

**Fitting** goes through `MotionClassifier`, which refuses below `MIN_TRACKS_TO_TRAIN`.
That refusal is the correct output at n=10, not a failure to work around: a
gradient-boosted tree fitted on ten samples reports an accuracy that is an artefact of
the split, and grouped cross-validation over eight clips cannot form folds that mean
anything.

**Separation** is descriptive and always runs. Per feature, per class: the range each
class occupies and whether those ranges overlap at all. "Birds showed a vertical
oscillation between 3.1 and 6.2 Hz; the drones ranged 0.1 to 0.4 Hz, and the ranges do
not overlap" is a real observation at n=3 vs n=5 and worth recording as preliminary. It
is **not** a p-value, an effect size, or evidence that the separation generalises, and
`SeparationReport` says so in its own output rather than leaving it to a reader.

Non-overlap at these counts happens easily by chance. With 3 and 5 samples drawn from one
distribution the two ranges are disjoint with probability 1 in 28
`[VERIFIED: 2 / C(8,3) = 2/56 = 1/28]`, so across nine features expect 0.32 such
separations from that pair alone. Over the three pairs of a bird/drone/aircraft split at
(3, 5, 2) it is **2.98** `[VERIFIED: 9 * (2/C(8,3) + 2/C(5,3) + 2/C(7,5)) = 2.98]` - which
is to say that finding two or three cleanly separated features at this sample size is what
noise looks like, not what a signal looks like. `expected_by_chance` computes this for the
actual class sizes and the report prints it next to whatever it found.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from tayr.classify.motion import MotionClassifier, NotEnoughDataError
from tayr.classify.motion import TrainingReport as MotionTrainingReport
from tayr.errors import TayrError
from tayr.tracking.features import TrackFeatures
from tayr.tracks.store import LabelledTrack


@dataclass(frozen=True, slots=True)
class ClassRange:
    """One class's spread on one feature."""

    label: str
    n: int
    low: float
    median: float
    high: float

    def overlaps(self, other: ClassRange) -> bool:
        return self.low <= other.high and other.low <= self.high


@dataclass(frozen=True, slots=True)
class FeatureSeparation:
    """How one feature distributes across the classes."""

    feature: str
    ranges: tuple[ClassRange, ...]

    @property
    def separated_pairs(self) -> tuple[tuple[str, str], ...]:
        """Class pairs whose observed ranges do not overlap at all."""
        pairs: list[tuple[str, str]] = []
        for i, a in enumerate(self.ranges):
            for b in self.ranges[i + 1 :]:
                if not a.overlaps(b):
                    pairs.append((a.label, b.label))
        return tuple(pairs)


def chance_separation_probability(n_a: int, n_b: int) -> float:
    """P(two samples' ranges do not overlap | they came from one distribution).

    Under exchangeability every interleaving of the ranks is equally likely, so the two
    arrangements that keep the groups contiguous - all of A below all of B, or the
    reverse - give 2 / C(n_a + n_b, n_a).
    """
    if n_a < 1 or n_b < 1:
        return 1.0
    return 2.0 / math.comb(n_a + n_b, n_a)


@dataclass(slots=True)
class SeparationReport:
    """Descriptive separation, with the small-sample caveat attached to it."""

    features: list[FeatureSeparation] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    n_features_examined: int = 0

    @property
    def separating(self) -> list[FeatureSeparation]:
        return [f for f in self.features if f.separated_pairs]

    def expected_by_chance(self) -> float:
        """How many clean separations to expect from noise, across all features.

        The number that stops a single non-overlapping feature being read as a finding.
        """
        labels = sorted(self.class_counts)
        per_feature = 0.0
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                per_feature += chance_separation_probability(
                    self.class_counts[a], self.class_counts[b]
                )
        return per_feature * self.n_features_examined

    def render(self) -> str:
        lines = [
            "Feature separation (DESCRIPTIVE, not inferential)",
            "=" * 60,
            "  Observed ranges per class. This is not a significance test and no p-value",
            "  is computed: at these counts none would be meaningful.",
            "",
        ]
        for sep in self.features:
            marks = ", ".join(f"{a} vs {b}" for a, b in sep.separated_pairs)
            flag = f"   <- ranges disjoint: {marks}" if marks else ""
            lines.append(f"  {sep.feature}{flag}")
            for r in sep.ranges:
                lines.append(
                    f"      {r.label:10s} n={r.n:<3d} [{r.low:>10.4f} .. {r.high:>10.4f}]  "
                    f"median {r.median:>10.4f}"
                )
        expected = self.expected_by_chance()
        found = len(self.separating)
        lines += [
            "",
            f"  {found} feature(s) show disjoint ranges. From noise alone, with these class sizes,",
            f"  about {expected:.1f} would be expected across {self.n_features_examined} features.",
        ]
        if found <= expected:
            lines.append(
                "  THAT IS AT OR BELOW CHANCE. Nothing here is evidence of a real separation."
            )
        else:
            lines.append(
                "  Above chance, but this is a range comparison on a handful of tracks - "
                "record it\n  as preliminary and worth collecting more data to test, "
                "never as a result."
            )
        return "\n".join(lines)


@dataclass(slots=True)
class FitOutcome:
    """What happened when the classifier was asked to train on what exists."""

    trained: bool
    n_tracks: int
    n_groups: int
    class_counts: dict[str, int]
    report: MotionTrainingReport | None = None
    refusal: str | None = None
    """Why training did not happen. A refusal is a result, not an error."""

    def render(self) -> str:
        lines = ["Motion classifier", "=" * 60]
        lines.append(
            f"  {self.n_tracks} track(s) from {self.n_groups} clip(s): "
            + ", ".join(f"{c}={n}" for c, n in sorted(self.class_counts.items()))
        )
        if not self.trained:
            lines += ["", "  NOT TRAINED.", f"  {self.refusal}"]
            lines.append(
                "  This is the correct outcome at this sample size, not a bug to work "
                "around. A\n  model fitted here would report an accuracy that is an "
                "artefact of the split."
            )
            return "\n".join(lines)

        if self.report is None:  # pragma: no cover - trained implies a report
            raise TayrError("FitOutcome says trained but carries no report")
        lines.append(
            f"  trained on {self.report.n_tracks} track(s), {self.report.n_groups} group(s)"
        )
        lines += ["", "  permutation importance:"]
        for name, value in sorted(self.report.feature_importances.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {name:28s} {value:>8.4f}")
        if self.report.warnings:
            lines += ["", "  WARNINGS:"] + [f"    - {w}" for w in self.report.warnings]
        return "\n".join(lines)


def separate(tracks: list[LabelledTrack]) -> SeparationReport:
    """Per-feature ranges by class. Always available, however few tracks there are."""
    report = SeparationReport(class_counts=dict(Counter(t.label for t in tracks)))
    if not tracks:
        return report

    names = TrackFeatures.feature_names()
    report.n_features_examined = len(names)
    by_label: dict[str, list[LabelledTrack]] = {}
    for track in tracks:
        by_label.setdefault(track.label, []).append(track)

    for name in names:
        ranges: list[ClassRange] = []
        for label in sorted(by_label):
            values = sorted(t.features[name] for t in by_label[label])
            ranges.append(
                ClassRange(
                    label=label,
                    n=len(values),
                    low=values[0],
                    median=values[len(values) // 2],
                    high=values[-1],
                )
            )
        report.features.append(FeatureSeparation(feature=name, ranges=tuple(ranges)))
    return report


def fit_motion_arm(tracks: list[LabelledTrack], *, seed: int = 1337) -> FitOutcome:
    """Train the baseline on whatever exists, or say precisely why it did not.

    `NotEnoughDataError` is caught and reported rather than raised: "there are ten tracks
    and ten is not enough" is the answer to the question being asked, and a traceback
    would make a correct refusal look like a broken command.
    """
    counts = dict(Counter(t.label for t in tracks))
    groups = {t.source_clip for t in tracks}

    if not tracks:
        return FitOutcome(
            trained=False,
            n_tracks=0,
            n_groups=0,
            class_counts={},
            refusal="the store is empty; ingest some runs first.",
        )

    classifier = MotionClassifier(seed=seed)
    try:
        report = classifier.fit(
            [t.to_features() for t in tracks],
            [t.label for t in tracks],
            [t.source_clip for t in tracks],
        )
    except NotEnoughDataError as exc:
        return FitOutcome(
            trained=False,
            n_tracks=len(tracks),
            n_groups=len(groups),
            class_counts=counts,
            # The exception already names the minimum; repeating it here printed it
            # twice in the one message an operator actually reads.
            refusal=str(exc),
        )

    return FitOutcome(
        trained=True,
        n_tracks=len(tracks),
        n_groups=len(groups),
        class_counts=counts,
        report=report,
    )


def hypothesis_status(tracks: list[LabelledTrack], outcome: FitOutcome) -> str:
    """What the current data licenses saying about the motion hypothesis.

    The hypothesis is a COMPARISON - motion against appearance, on the same tracks, in
    the same pixels-on-target buckets. Neither arm can run yet, and saying so is the
    whole output. `compare_arms` is not invoked with empty arms, because a
    `HypothesisResult` built from nothing renders like a result that came out
    undetermined rather than one that was never computed.
    """
    lines = ["Hypothesis: does motion separate drones from birds below ~20px?", "=" * 60]
    if not outcome.trained:
        lines += [
            "  UNDETERMINED - and not because the intervals overlapped.",
            "",
            "  The motion arm did not train, so no arm was evaluated and no comparison",
            "  exists. The appearance arm needs a fitted CNN over track crops, which also",
            "  does not exist. Two missing arms is not a null result; it is an experiment",
            "  that has not been run.",
        ]
    else:
        lines += [
            "  The motion arm trained. The comparison still cannot run: the appearance",
            "  arm needs a fitted CNN over track crops and there is none, so there is",
            "  nothing to compare against. UNDETERMINED, for want of the other arm.",
        ]

    buckets = Counter("<=20px" if t.median_pixels_on_target <= 20.0 else ">20px" for t in tracks)
    lines += [
        "",
        "  tracks by the size boundary the hypothesis is about:",
        *(f"    {k:10s} {v}" for k, v in sorted(buckets.items())),
    ]
    if buckets.get("<=20px", 0) == 0:
        lines.append(
            "    NONE below 20px. The hypothesis is specifically about small targets, so "
            "this\n    data cannot address it at all, whatever the classifier does."
        )
    return "\n".join(lines)
