"""What a tool is allowed to reach.

`ToolContext` is the whole surface available to a tool handler. Anything not on it is
unreachable, which makes least privilege a property of this dataclass rather than of
each handler behaving.

Deliberately absent, and each for a reason:

  no HTTP client         tools do not reach the public internet; the worker network has
                         no egress and the agent must not need it
  no database session    read-only tools are given already-loaded track data, so a tool
                         cannot issue an arbitrary query on the model's behalf
  no filesystem root     only the evidence directory, and clips are written by path
                         construction the model never influences
  no shell               nothing here can execute a command
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tayr.agent.site_config import SiteRegistry
from tayr.tracking.features import TrackFeatures


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    """One track, already loaded. Tools read this rather than querying.

    Passing loaded data instead of a session means a tool cannot be talked into reading
    a row it was not given, whatever arguments the model invents.
    """

    track_id: str
    job_id: str
    track_number: int
    first_frame: int
    last_frame: int
    n_observations: int
    median_pixels_on_target: float
    fps: float
    features: TrackFeatures | None = None
    """None when the track was too short for motion features to mean anything."""

    classifier_label: str = "unknown"
    classifier_confidence: float = 0.0
    classifier_interval: tuple[float, float] | None = None
    """Wilson interval from the classifier. None when no classifier is trained, which
    is the current state of every deployment."""

    classifier_trained: bool = False
    observed_at: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return (self.last_frame - self.first_frame + 1) / self.fps


@dataclass(frozen=True, slots=True)
class HistoricalTrack:
    """A prior track at the same site, for repeat-pattern lookup."""

    track_id: str
    observed_at: datetime
    median_pixels_on_target: float
    verdict: str
    heading_entropy: float | None = None


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Everything the registered tools may reach. Nothing else is available."""

    site_id: str
    tracks: dict[str, TrackSnapshot]
    """Keyed by track_id. A tool that is handed a track_id not in here fails - which is
    how "the track must belong to this job" is enforced without a query."""

    registry: SiteRegistry
    evidence_dir: Path
    now: datetime
    history: tuple[HistoricalTrack, ...] = ()
    synthetic: bool = False
    """True when any input came from a placeholder detector or classifier."""

    notifier: object | None = None
    """Set only when acting tools are registered. Read-only runs leave it None."""

    incident_store: object | None = None
    action_budget: dict[str, int] = field(default_factory=dict)
