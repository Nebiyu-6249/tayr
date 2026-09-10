"""Turn a `tayr watch run` output directory into labelled track rows.

The features are already there. `analyze_track` records the whole motion vector into
every decision, so a run directory is a labelled dataset waiting for one thing: somebody
to say what the clip contained.

## What is read, and what is refused

Only `decisions.json`, and only tracks that carry features. A track with fewer
observations than the feature extractor needs has `features_available: false` in its
record, and admitting it with zeros would put fabricated rows into the one dataset this
project has. Those are counted and reported, never silently dropped.

Tracks shorter than `MIN_TRACK_SECONDS` are also excluded, because that is the threshold
the verdict rules already apply before characterising motion. A dataset holding tracks
the rules would refuse to judge would be measuring something the system never does.

## Rounding

The record stores features rounded to 3-5 decimal places. That is a real precision loss
against the in-memory values and it is recorded in the row's notes. It is immaterial to a
gradient-boosted tree, which splits on ordering rather than on magnitude, and the
alternative - re-running the pipeline to recover full precision - costs minutes per clip
to recover digits no split will ever see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tayr.errors import ConfigError
from tayr.tracks.store import MIN_TRACK_SECONDS, LabelBasis, LabelledTrack

#: Feature keys in the record, mapped to `TrackFeatures` field names. The record uses
#: `mean_speed_px_per_frame` where the dataclass uses `mean_speed`; every other name
#: matches. Written out rather than derived so a rename upstream fails loudly here.
_FEATURE_KEYS: dict[str, str] = {
    "mean_speed_px_per_frame": "mean_speed",
    "speed_variance": "speed_variance",
    "acceleration_variance": "acceleration_variance",
    "vertical_oscillation_hz": "vertical_oscillation_hz",
    "vertical_oscillation_power": "vertical_oscillation_power",
    "heading_entropy": "heading_entropy",
    "hover_fraction": "hover_fraction",
    "trajectory_smoothness": "trajectory_smoothness",
    "scale_change_rate": "scale_change_rate",
}

_ROUNDING_NOTE = "features read from a decision record, which rounds them to 3-5 decimal places"


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What a run directory yielded, and what it did not."""

    tracks: list[LabelledTrack]
    n_decisions: int
    n_without_features: int
    n_too_short: int
    source_clip: str
    notes: list[str]

    @property
    def n_ingested(self) -> int:
        return len(self.tracks)

    def render(self) -> str:
        lines = [
            f"{self.source_clip}: {self.n_ingested} of {self.n_decisions} track(s) ingested",
        ]
        if self.n_without_features:
            lines.append(
                f"  {self.n_without_features} had no motion features (too few "
                "observations) and were excluded rather than zero-filled"
            )
        if self.n_too_short:
            lines.append(
                f"  {self.n_too_short} were shorter than {MIN_TRACK_SECONDS:g}s, the "
                "threshold the verdict rules use before characterising motion"
            )
        lines += [f"  {note}" for note in self.notes]
        return "\n".join(lines)


def source_clip_of(run_dir: Path) -> str:
    """The clip a run came from, taken from its manifest, else the directory name.

    This is the split group key, so getting it wrong silently breaks the one guarantee
    the dataset makes about independence. The manifest records the config, not the video
    path, so the directory name is the honest fallback and the caller can override it.
    """
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        try:
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return run_dir.name
        run_id = recorded.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    return run_dir.name


def ingest_run(
    run_dir: Path,
    *,
    label: str,
    asserted_by: str,
    source_clip: str | None = None,
    min_seconds: float = MIN_TRACK_SECONDS,
) -> IngestReport:
    """Read one run directory into labelled rows.

    `label` is what the operator says the clip contains. It is applied to every track
    from the clip, which is exactly why `LabelBasis.CLIP_PROVENANCE` is recorded
    alongside it: nobody has looked at these tracks individually.
    """
    decisions_path = run_dir / "decisions.json"
    if not decisions_path.is_file():
        raise ConfigError(
            f"no decisions.json in {run_dir}. Point this at a `tayr watch run` output "
            "directory, not at the video."
        )
    try:
        records = json.loads(decisions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{decisions_path} is not valid JSON: {exc}") from exc
    if not isinstance(records, list):
        raise ConfigError(f"{decisions_path} must hold a list of decisions.")

    clip = source_clip or source_clip_of(run_dir)
    run_id = str(records[0].get("job_id", clip)) if records else clip

    tracks: list[LabelledTrack] = []
    without_features = 0
    too_short = 0
    notes: list[str] = []
    synthetic = any(bool(r.get("synthetic")) for r in records)
    if synthetic:
        notes.append(
            "SYNTHETIC RUN. At least one decision here came from a stand-in detector, so "
            "these tracks describe placeholder input and must not enter a dataset used "
            "for a reported result."
        )

    for record in records:
        analysis = _analysis_of(record)
        if analysis is None:
            without_features += 1
            continue
        if not analysis.get("features_available"):
            without_features += 1
            continue
        duration = float(analysis.get("duration_seconds", 0.0))
        if duration < min_seconds:
            too_short += 1
            continue

        raw = analysis.get("features") or {}
        try:
            features = {field: float(raw[key]) for key, field in _FEATURE_KEYS.items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(
                f"{decisions_path}: track {record.get('track_id')!r} has an "
                f"analyze_track payload this ingester does not understand ({exc}). The "
                "tool's feature keys changed; update _FEATURE_KEYS rather than guessing."
            ) from exc

        tracks.append(
            LabelledTrack(
                track_id=str(record["track_id"]),
                source_clip=clip,
                label=label,
                # Never OPERATOR_CONFIRMED from here: this function has a clip-level
                # assertion and nothing else. Only a human pressing a button on one
                # track can upgrade it.
                label_basis=LabelBasis.CLIP_PROVENANCE,
                asserted_by=asserted_by,
                run_id=run_id,
                duration_seconds=duration,
                n_observations=int(analysis.get("n_observations", 0)),
                median_pixels_on_target=float(analysis.get("median_pixels_on_target", 0.0)),
                features=features,
                notes=(_ROUNDING_NOTE,) + (("synthetic run",) if synthetic else ()),
            )
        )

    return IngestReport(
        tracks=tracks,
        n_decisions=len(records),
        n_without_features=without_features,
        n_too_short=too_short,
        source_clip=clip,
        notes=notes,
    )


def _analysis_of(record: dict[str, Any]) -> dict[str, Any] | None:
    """The `analyze_track` result from one decision, or None if it never ran."""
    for call in record.get("tool_calls", ()):
        if call.get("tool_name") == "analyze_track" and call.get("ok"):
            result = call.get("result")
            return result if isinstance(result, dict) else None
    return None
