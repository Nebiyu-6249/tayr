"""Labelled track data, collected from runs over footage the operator can identify.

This is the answer to the gap `docs/RESEARCH.md` 14.4 has recorded since Phase 0: no
identified source supplies a labelled bird track, so the motion hypothesis could not be
tested. Running Tayr over footage somebody can name produces tracks whose motion features
are already computed and whose class the clip supplies.

The label is provenance - which clip, and what a person said it contains - and never a
per-track annotation. See `store.LabelBasis`.
"""

from tayr.tracks.analysis import (
    FitOutcome,
    SeparationReport,
    fit_motion_arm,
    hypothesis_status,
    separate,
)
from tayr.tracks.ingest import IngestReport, ingest_run
from tayr.tracks.store import (
    MIN_TRACK_SECONDS,
    MIN_TRACKS_PER_CLASS,
    LabelBasis,
    LabelledTrack,
    TrackCensus,
    TrackStore,
)

__all__ = [
    "MIN_TRACKS_PER_CLASS",
    "MIN_TRACK_SECONDS",
    "FitOutcome",
    "IngestReport",
    "LabelBasis",
    "LabelledTrack",
    "SeparationReport",
    "TrackCensus",
    "TrackStore",
    "fit_motion_arm",
    "hypothesis_status",
    "ingest_run",
    "separate",
]
