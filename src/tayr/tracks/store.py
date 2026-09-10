"""A labelled track dataset, built from runs the operator has watched.

## Why this exists

`docs/RESEARCH.md` §14.4 has said since Phase 0 that the motion hypothesis cannot be
tested, because no identified source supplies a single labelled **bird track**. Detection
datasets supply boxes; tracking datasets supply drones. Neither supplies a bird moving
through a frame with its motion recorded.

Running Tayr over footage the operator can identify produces exactly that. Every track
already carries the full motion feature vector, and the clip supplies the class. This
turns "we cannot test the hypothesis" into "we can test it at n=10, which is
under-powered and says so".

## The label is provenance, not ground truth

**This is the most important property in this module.** A row's label records two facts:
which clip the track came from, and what a person said that clip contains. It does not
record that anyone looked at that track.

A clip of seagulls can have an aircraft in shot. A drone clip can produce a track on a
bird that wandered through. `LabelBasis` carries this so that no later reader mistakes a
clip-level assertion for a per-track annotation - and so that when someone does verify a
track by eye, the upgrade is recorded rather than assumed.

## Grouping

`source_clip` is the group key, and every split must use it. Two tracks from one clip
share a sky, a camera, a compression history and often the same animal; treating them as
independent samples inflates every number computed from them. This is the same rule as
`EvalConfig.group_splits_by_video` and it exists for the same reason.

## Storage

JSON Lines outside the repository. Append-only, one row per track, and never committed:
these rows are derived from footage whose licence this project does not control, and
`.gitignore` plus the `licence-guard` CI job both exist because intent is not a control.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from tayr.errors import ConfigError
from tayr.tracking.features import TrackFeatures

#: Tracks per class below which the census warns. Matches `_MIN_TRACKS_PER_CLASS` in
#: `datasets/census.py` deliberately: one target, stated in one place per subsystem, and
#: the same number a reader has already seen quoted about datasets.
MIN_TRACKS_PER_CLASS = 50

#: Shortest track that can carry a motion characterisation. Matches
#: `MIN_TRACK_SECONDS_TO_JUDGE` in `agent/rules.py` - the threshold the verdict rules
#: already apply - so the dataset holds what the rules would actually act on.
MIN_TRACK_SECONDS = 2.0


class LabelBasis(StrEnum):
    """How much is actually known about a row's label."""

    CLIP_PROVENANCE = "clip_provenance"
    """An operator said this clip contains this class. Nobody inspected this track.

    The honest default and the only basis this module can produce on its own. A clip of
    birds can contain an aircraft; a track from it inherits `bird` regardless.
    """

    OPERATOR_CONFIRMED = "operator_confirmed"
    """A person looked at this specific track and confirmed the class.

    Never set by ingestion. Reserved for the Slack feedback loop, where a human presses a
    button against one track - which is the only place a per-track label can come from.
    """


@dataclass(frozen=True, slots=True)
class LabelledTrack:
    """One track, its motion features, and where its label came from."""

    track_id: str
    source_clip: str
    """Group key for every split. Two tracks from one clip are not independent."""

    label: str
    label_basis: LabelBasis
    asserted_by: str
    """Who said the clip contains this class. Recorded so a disputed row has an author."""

    run_id: str
    duration_seconds: float
    n_observations: int
    median_pixels_on_target: float
    features: dict[str, float]
    ingested_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: tuple[str, ...] = ()

    def to_features(self) -> TrackFeatures:
        """Rebuild the vector the classifier consumes.

        `duration_frames` is not carried by a decision record and is not part of
        `TrackFeatures.as_vector()`, so it is stored as 0 rather than reconstructed from
        a frame rate the record does not contain. Deriving it would put an invented
        number into a dataset whose whole value is that its provenance is traceable.
        """
        return TrackFeatures(
            n_observations=self.n_observations,
            duration_frames=0,
            mean_speed=self.features["mean_speed"],
            speed_variance=self.features["speed_variance"],
            acceleration_variance=self.features["acceleration_variance"],
            vertical_oscillation_hz=self.features["vertical_oscillation_hz"],
            vertical_oscillation_power=self.features["vertical_oscillation_power"],
            heading_entropy=self.features["heading_entropy"],
            hover_fraction=self.features["hover_fraction"],
            trajectory_smoothness=self.features["trajectory_smoothness"],
            scale_change_rate=self.features["scale_change_rate"],
            median_pixels_on_target=self.median_pixels_on_target,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


@dataclass(frozen=True, slots=True)
class TrackCensus:
    """How much labelled track data exists, and how far from enough."""

    counts: dict[str, int]
    clips_per_class: dict[str, int]
    total: int
    store_path: Path

    def shortfall(self, target: int = MIN_TRACKS_PER_CLASS) -> dict[str, int]:
        """Tracks still needed per class to reach `target`."""
        return {cls: max(0, target - n) for cls, n in sorted(self.counts.items()) if target - n > 0}

    def render(self, *, target: int = MIN_TRACKS_PER_CLASS) -> str:
        lines = [
            f"Labelled tracks: {self.total} in {self.store_path}",
            "=" * 60,
        ]
        if not self.counts:
            lines.append("  (empty)")
            return "\n".join(lines)

        for cls in sorted(self.counts):
            n, clips = self.counts[cls], self.clips_per_class.get(cls, 0)
            bar = "#" * min(40, round(40 * n / target)) if target else ""
            lines.append(f"  {cls:10s} {n:>4d} track(s) from {clips:>3d} clip(s)  {bar}")

        lines += ["", f"  toward n={target} per class:"]
        shortfall = self.shortfall(target)
        if not shortfall:
            lines.append(f"    every class has at least {target}.")
        else:
            for cls, missing in shortfall.items():
                lines.append(f"    {cls:10s} needs {missing:>4d} more")

        # Clips, not tracks, are the unit of independence - so they are the unit of the
        # honest progress number. Ten tracks from two clips is nearer to two samples.
        thin = [cls for cls, clips in self.clips_per_class.items() if clips < 3]
        if thin:
            lines += [
                "",
                f"  FEW SOURCE CLIPS for {', '.join(sorted(thin))}. Splits group by clip, "
                "so a class from one or two clips cannot be cross-validated at all: "
                "every fold either holds all of it or none of it.",
            ]
        return "\n".join(lines)


class TrackStore:
    """Append-only JSON Lines of labelled tracks, held outside the repository."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[LabelledTrack]:
        if not self.path.is_file():
            return []
        rows: list[LabelledTrack] = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{self.path}:{number} is not valid JSON: {exc}") from exc
            rows.append(_row_from(raw, where=f"{self.path}:{number}"))
        return rows

    def append(self, tracks: list[LabelledTrack]) -> int:
        """Add rows, skipping any `track_id` already stored. Returns how many were new.

        Re-ingesting a run is a normal accident - the same directory named twice - and
        silently duplicating its tracks would double-count them in every census and put
        near-identical rows on both sides of a split.
        """
        known = {row.track_id for row in self.load()}
        fresh = [t for t in tracks if t.track_id not in known]
        if not fresh:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for row in fresh:
                handle.write(row.to_json() + "\n")
        return len(fresh)

    def census(self) -> TrackCensus:
        rows = self.load()
        counts = Counter(row.label for row in rows)
        clips: dict[str, set[str]] = {}
        for row in rows:
            clips.setdefault(row.label, set()).add(row.source_clip)
        return TrackCensus(
            counts=dict(counts),
            clips_per_class={cls: len(seen) for cls, seen in clips.items()},
            total=len(rows),
            store_path=self.path,
        )


def _row_from(raw: dict[str, Any], *, where: str) -> LabelledTrack:
    try:
        return LabelledTrack(
            track_id=str(raw["track_id"]),
            source_clip=str(raw["source_clip"]),
            label=str(raw["label"]),
            label_basis=LabelBasis(raw["label_basis"]),
            asserted_by=str(raw["asserted_by"]),
            run_id=str(raw["run_id"]),
            duration_seconds=float(raw["duration_seconds"]),
            n_observations=int(raw["n_observations"]),
            median_pixels_on_target=float(raw["median_pixels_on_target"]),
            features=dict(raw["features"]),
            ingested_at=str(raw.get("ingested_at", "")),
            notes=tuple(raw.get("notes", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{where} is not a labelled track row: {exc}") from exc
