"""Track census.

This is the first thing to run on any dataset, before writing a line of training code.

The motion arm of Tayr's hypothesis classifies **tracks**, not frames. A dataset
advertising "10,000 images" may contain zero tracks, and a 20-video tracking subset
contains roughly 20 tracks - which is a very different quantity from 20,000 boxes when
what you are training is a per-track classifier.

The census exists to answer one question early enough to act on it:

    How many usable tracks are there, per class, and how long are they?

If the answer for birds is small, the hypothesis test as specified is not adequately
powered, and the project's framing has to change. That is a week-two discovery, not a
week-twenty one. See docs/RESEARCH.md R2.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from tayr.datasets.schema import DatasetAnnotation, ObjectClass, TrackIdSource
from tayr.geometry import SizeBucket, pixels_on_target, size_bucket


@dataclass(frozen=True, slots=True)
class TrackSummary:
    """One track: a (source_video, track_id) pair observed across frames."""

    source_video: str
    track_id: int
    label: ObjectClass
    n_observations: int
    first_frame: int
    last_frame: int
    median_pixels_on_target: float

    @property
    def span(self) -> int:
        """Frames from first to last observation, gaps included."""
        return self.last_frame - self.first_frame + 1


@dataclass(slots=True)
class Census:
    """What a dataset actually contains, measured rather than quoted."""

    dataset: str
    split: str
    n_videos: int = 0
    n_frames: int = 0
    n_empty_frames: int = 0
    n_boxes: int = 0
    boxes_per_class: dict[str, int] = field(default_factory=dict)
    tracks_per_class: dict[str, int] = field(default_factory=dict)
    track_id_sources: dict[str, int] = field(default_factory=dict)
    size_buckets: dict[str, int] = field(default_factory=dict)
    tracks: list[TrackSummary] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    def render(self) -> str:
        """Human-readable report. This is what gets pasted into a progress update."""
        lines = [
            f"Census: {self.dataset} [{self.split}]",
            "=" * 60,
            f"  videos           {self.n_videos}",
            f"  frames           {self.n_frames}  (empty: {self.n_empty_frames})",
            f"  boxes            {self.n_boxes}",
            f"  tracks           {self.n_tracks}",
            "",
            "  boxes per class:",
        ]
        for cls in sorted(self.boxes_per_class):
            lines.append(f"    {cls:12s} {self.boxes_per_class[cls]:>8d}")

        lines += ["", "  TRACKS per class  <- the unit the motion classifier trains on:"]
        for cls in sorted(ObjectClass, key=lambda c: c.value):
            n = self.tracks_per_class.get(cls.value, 0)
            lines.append(f"    {cls.value:12s} {n:>8d}")

        lines += ["", "  track id provenance:"]
        for src in sorted(self.track_id_sources):
            lines.append(f"    {src:14s} {self.track_id_sources[src]:>6d} video(s)")

        lines += ["", "  pixels-on-target distribution:"]
        for bucket in (SizeBucket.TINY, SizeBucket.SMALL, SizeBucket.MEDIUM, SizeBucket.LARGE):
            n = self.size_buckets.get(bucket.value, 0)
            pct = (100.0 * n / self.n_boxes) if self.n_boxes else 0.0
            lines.append(f"    {bucket.value:10s} {n:>8d}  ({pct:5.1f}%)")

        if self.tracks:
            obs = np.array([t.n_observations for t in self.tracks])
            lines += [
                "",
                "  track length (observations):",
                f"    min {obs.min()}  median {int(np.median(obs))}  max {obs.max()}",
            ]

        if self.warnings:
            lines += ["", "  WARNINGS:"]
            lines += [f"    - {w}" for w in self.warnings]

        return "\n".join(lines)


def take_census(dataset: DatasetAnnotation) -> Census:
    """Measure a dataset. Every number here comes from counting, not from a paper."""
    census = Census(dataset=dataset.name, split=dataset.split)

    box_classes: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    id_sources: Counter[str] = Counter()

    # (video, track_id) -> observations
    grouped: dict[tuple[str, int], list[tuple[int, ObjectClass, float]]] = defaultdict(list)
    untracked_boxes = 0

    for video in dataset.videos:
        census.n_videos += 1
        census.n_frames += len(video.frames)
        census.n_empty_frames += video.n_empty_frames
        id_sources[video.track_id_source.value] += 1

        for frame in video.frames:
            for obj in frame.objects:
                census.n_boxes += 1
                box_classes[obj.label.value] += 1

                arr = obj.as_array()
                buckets[str(size_bucket(arr).item())] += 1
                pot = float(pixels_on_target(arr))

                if obj.track_id is None:
                    untracked_boxes += 1
                else:
                    grouped[(video.source_video, obj.track_id)].append(
                        (frame.frame_index, obj.label, pot)
                    )

    for (source_video, track_id), observations in sorted(grouped.items()):
        frames = [f for f, _, _ in observations]
        labels = Counter(lbl for _, lbl, _ in observations)
        pots = [p for _, _, p in observations]

        # A track whose label changes mid-way is an annotation or association bug.
        # Take the majority label and say so, rather than picking silently.
        label, _ = labels.most_common(1)[0]
        if len(labels) > 1:
            census.warnings.append(
                f"track {source_video}#{track_id} carries {len(labels)} different class "
                f"labels {dict(labels)}; using the majority ({label.value})."
            )

        census.tracks.append(
            TrackSummary(
                source_video=source_video,
                track_id=track_id,
                label=label,
                n_observations=len(observations),
                first_frame=min(frames),
                last_frame=max(frames),
                median_pixels_on_target=float(np.median(pots)),
            )
        )

    census.boxes_per_class = dict(box_classes)
    census.size_buckets = dict(buckets)
    census.track_id_sources = dict(id_sources)
    census.tracks_per_class = dict(Counter(t.label.value for t in census.tracks))

    _add_warnings(census, untracked_boxes)
    return census


# Below this many tracks in a class, a per-class result is not worth reporting as a
# point estimate. Chosen as an order-of-magnitude tripwire, not a power calculation -
# the real interval comes from the evaluation itself.
_MIN_TRACKS_PER_CLASS = 50


def _add_warnings(census: Census, untracked_boxes: int) -> None:
    if untracked_boxes:
        census.warnings.append(
            f"{untracked_boxes} box(es) have no track identity and cannot be used by "
            "the motion arm. They still count toward detection training."
        )

    if census.n_tracks == 0 and census.n_boxes > 0:
        census.warnings.append(
            "ZERO tracks. This dataset can train a detector but contributes nothing to "
            "the motion hypothesis, which classifies tracks."
        )

    inferred = census.track_id_sources.get(TrackIdSource.SINGLE_TARGET.value, 0)
    if inferred:
        census.warnings.append(
            f"{inferred} video(s) had track identity INFERRED from a single-target "
            "assumption rather than annotated. Any per-track count derived from them "
            "rests on that assumption."
        )

    for cls in (ObjectClass.DRONE, ObjectClass.BIRD, ObjectClass.AIRCRAFT):
        n = census.tracks_per_class.get(cls.value, 0)
        if n < _MIN_TRACKS_PER_CLASS:
            census.warnings.append(
                f"only {n} {cls.value} track(s) (< {_MIN_TRACKS_PER_CLASS}). A per-class "
                "result from this few tracks needs a confidence interval, and a point "
                "estimate alone would be an overclaim."
            )
