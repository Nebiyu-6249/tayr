"""The read-only tools.

None of these change anything. Each takes a Pydantic-validated argument model with
explicit bounds, and each returns a plain dict that goes into the audit record verbatim.

The recurring pattern in every handler: **a track_id that is not in the context fails.**
The context holds only tracks from the job under evaluation, so "the track must exist
and belong to this job" is enforced by lookup rather than by a check someone can forget
to write.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from tayr.agent.site_config import AuthorizationMatch, ZoneStatus
from tayr.agent.tools.context import ToolContext
from tayr.agent.tools.registry import ToolEffect, ToolSpec
from tayr.errors import TayrError

# The motion bands the rules and the prose both refer to. Defined once, here, so a
# threshold quoted to an operator is the same number the verdict used.
BIRD_FLAP_HZ_LOW = 2.0
BIRD_FLAP_HZ_HIGH = 8.0
"""Wingbeat frequencies for birds of a size detectable at range. `[ASSUMED]` - this band
is a design parameter, not a measured result. Tayr has no bird tracks to fit it against
(docs/RESEARCH.md 14.4), so it is stated as an assumption everywhere it is used and must
be validated before any claim rests on it."""

HOVER_SPEED_PX_PER_FRAME = 0.5
MIN_HOVER_SECONDS = 3.0
"""Sustained station-keeping beyond this is not bird flight. Also `[ASSUMED]`."""

SMALL_TARGET_PX = 20.0
"""Below this, appearance classification is unreliable and a verdict must be
motion-based. This threshold is the project's research hypothesis, not a finding."""


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


TrackId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")]
"""Bounded and character-restricted. A track_id is a uuid in practice; anything that is
not plausibly one is rejected before it reaches a lookup."""


class AnalyzeTrackArgs(_Args):
    track_id: TrackId


class CheckAuthorizationArgs(_Args):
    track_id: TrackId


class CheckAirspaceZoneArgs(_Args):
    site_id: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")]
    timestamp: datetime | None = None


class QueryHistoryArgs(_Args):
    """Concrete, because `query_history(pattern)` as specified was too loose to validate.

    A free-text "pattern" would be an unvalidatable string reaching a lookup. This asks
    the question the brief actually wanted - has something like this been seen here
    before - in bounded terms.
    """

    size_band_px: Annotated[float, Field(gt=0.0, le=10_000.0)]
    tolerance_px: Annotated[float, Field(default=8.0, ge=0.0, le=100.0)]
    window_days: Annotated[int, Field(default=30, ge=1, le=365)]


class GetEvidenceClipArgs(_Args):
    track_id: TrackId
    pre_seconds: Annotated[float, Field(default=3.0, ge=0.0, le=30.0)]
    post_seconds: Annotated[float, Field(default=3.0, ge=0.0, le=30.0)]


def _require_track(context: ToolContext, track_id: str) -> Any:
    snapshot = context.tracks.get(track_id)
    if snapshot is None:
        raise TayrError(
            f"track {track_id!r} is not part of this job. Only tracks from the job under "
            "evaluation can be inspected."
        )
    return snapshot


def analyze_track(context: ToolContext, args: AnalyzeTrackArgs) -> dict[str, Any]:
    """Motion features and the classifier's verdict, with its interval."""
    track = _require_track(context, args.track_id)

    payload: dict[str, Any] = {
        "track_id": track.track_id,
        "track_number": track.track_number,
        "n_observations": track.n_observations,
        "duration_seconds": round(track.duration_seconds, 2),
        "median_pixels_on_target": round(track.median_pixels_on_target, 1),
        "appearance_reliable": track.median_pixels_on_target >= SMALL_TARGET_PX,
        "small_target_threshold_px": SMALL_TARGET_PX,
        "synthetic": context.synthetic,
    }

    if track.features is None:
        payload["features_available"] = False
        payload["reason"] = (
            "track has too few observations for motion features; an acceleration "
            "variance or dominant frequency from this many samples would be noise"
        )
    else:
        f = track.features
        payload["features_available"] = True
        payload["features"] = {
            "mean_speed_px_per_frame": round(f.mean_speed, 3),
            "speed_variance": round(f.speed_variance, 4),
            "acceleration_variance": round(f.acceleration_variance, 4),
            "vertical_oscillation_hz": round(f.vertical_oscillation_hz, 3),
            "vertical_oscillation_power": round(f.vertical_oscillation_power, 3),
            "heading_entropy": round(f.heading_entropy, 3),
            "hover_fraction": round(f.hover_fraction, 3),
            "trajectory_smoothness": round(f.trajectory_smoothness, 3),
            "scale_change_rate": round(f.scale_change_rate, 5),
        }
        payload["hover_seconds"] = round(f.hover_fraction * track.duration_seconds, 2)
        payload["bird_flap_band_hz"] = [BIRD_FLAP_HZ_LOW, BIRD_FLAP_HZ_HIGH]

    if not track.classifier_trained:
        payload["classifier"] = {
            "status": "no_classifier_trained",
            "detail": (
                "no trained track classifier in this deployment, so neither appearance "
                "nor motion classification is available. This is not a low-confidence "
                "answer; it is the absence of one."
            ),
        }
    else:
        interval = track.classifier_interval
        undetermined = interval is not None and (interval[1] - interval[0]) > 0.30
        payload["classifier"] = {
            "status": "undetermined" if undetermined else "determined",
            "label": track.classifier_label,
            "confidence": round(track.classifier_confidence, 3),
            "interval": [round(interval[0], 3), round(interval[1], 3)] if interval else None,
        }
    return payload


def check_authorization(context: ToolContext, args: CheckAuthorizationArgs) -> dict[str, Any]:
    """Is there a filed, permitted flight covering this site and time?

    In a real deployment this is the largest source of dismissals. It is also the only
    tool that can produce a confident DISMISS with no trained classifier.
    """
    track = _require_track(context, args.track_id)
    moment = track.observed_at or context.now
    site = context.registry.get(context.site_id)

    if site is None:
        # Not an authorization. An unknown site cannot vouch for anything, and saying
        # "not authorized" would imply a registry that was actually consulted.
        return AuthorizationMatch(authorized=False, site_known=False, checked_at=moment).model_dump(
            mode="json"
        )

    for flight in site.authorized_flights:
        if flight.covers(moment):
            return AuthorizationMatch(
                authorized=True,
                flight_id=flight.flight_id,
                operator=flight.operator,
                checked_at=moment,
            ).model_dump(mode="json")

    return AuthorizationMatch(authorized=False, checked_at=moment).model_dump(mode="json")


def check_airspace_zone(context: ToolContext, args: CheckAirspaceZoneArgs) -> dict[str, Any]:
    """The protection posture at this place and time."""
    moment = args.timestamp or context.now
    site = context.registry.get(args.site_id)
    if site is None:
        return ZoneStatus(site_id=args.site_id, site_known=False).model_dump(mode="json")

    active = next((r for r in site.restrictions if r.covers(moment)), None)
    return ZoneStatus(
        site_id=site.site_id,
        site_known=True,
        zone_class=site.zone_class,
        restriction_active=active is not None,
        restriction_reason=active.reason if active else None,
    ).model_dump(mode="json")


def query_history(context: ToolContext, args: QueryHistoryArgs) -> dict[str, Any]:
    """Has something of this size been seen at this site recently?

    A repeat incursion is a different matter from a one-off, and this is the only tool
    that can tell the difference.
    """
    cutoff_days = args.window_days
    lo = args.size_band_px - args.tolerance_px
    hi = args.size_band_px + args.tolerance_px

    matches = [
        h
        for h in context.history
        if lo <= h.median_pixels_on_target <= hi
        and (context.now - h.observed_at).days <= cutoff_days
    ]
    return {
        "window_days": cutoff_days,
        "size_band_px": [round(lo, 1), round(hi, 1)],
        "match_count": len(matches),
        "matches": [
            {
                "track_id": m.track_id,
                "observed_at": m.observed_at.isoformat(),
                "median_pixels_on_target": round(m.median_pixels_on_target, 1),
                "prior_verdict": m.verdict,
            }
            for m in matches[:10]
        ],
        "truncated": len(matches) > 10,
    }


def get_evidence_clip(context: ToolContext, args: GetEvidenceClipArgs) -> dict[str, Any]:
    """Locate the annotated segment around a track, with a content hash.

    The path is constructed from the track id and the evidence directory. The model
    never supplies any part of it, so no argument it invents can point somewhere else.
    """
    track = _require_track(context, args.track_id)

    start = max(0.0, (track.first_frame / track.fps if track.fps > 0 else 0.0) - args.pre_seconds)
    end = (track.last_frame / track.fps if track.fps > 0 else 0.0) + args.post_seconds

    clip_path = context.evidence_dir / f"{track.track_id}.mp4"
    exists = clip_path.is_file()
    digest = hashlib.sha256(clip_path.read_bytes()).hexdigest() if exists else None

    return {
        "track_id": track.track_id,
        "clip_path": str(clip_path),
        "exists": exists,
        "sha256": digest,
        "start_seconds": round(start, 2),
        "end_seconds": round(end, 2),
        "synthetic": context.synthetic,
        # Honest: with no trained detector there is no annotated render to cut. The
        # agent still reports the segment bounds, and says the clip is absent rather
        # than implying evidence that does not exist.
        "detail": None if exists else "no rendered clip available for this track yet",
    }


READ_ONLY_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="analyze_track",
        description=(
            "Motion features for a track (oscillation frequency, hover fraction, "
            "heading entropy, acceleration variance, trajectory smoothness, "
            "pixels-on-target, duration) plus the track classifier's verdict and its "
            "confidence interval. Reports when no classifier is trained."
        ),
        args_model=AnalyzeTrackArgs,
        handler=analyze_track,  # type: ignore[arg-type]
        effect=ToolEffect.READ_ONLY,
    ),
    ToolSpec(
        name="check_authorization",
        description=(
            "Check whether a filed, permitted flight covers this site and the time the "
            "track was observed. Most detected drones are somebody's authorized flight."
        ),
        args_model=CheckAuthorizationArgs,
        handler=check_authorization,  # type: ignore[arg-type]
        effect=ToolEffect.READ_ONLY,
    ),
    ToolSpec(
        name="check_airspace_zone",
        description=(
            "The airspace posture at a site and time: zone classification and whether a "
            "temporary restriction is active."
        ),
        args_model=CheckAirspaceZoneArgs,
        handler=check_airspace_zone,  # type: ignore[arg-type]
        effect=ToolEffect.READ_ONLY,
    ),
    ToolSpec(
        name="query_history",
        description=(
            "Count prior tracks at this site within a size band and time window. A "
            "repeat incursion is a different matter from a one-off."
        ),
        args_model=QueryHistoryArgs,
        handler=query_history,  # type: ignore[arg-type]
        effect=ToolEffect.READ_ONLY,
    ),
    ToolSpec(
        name="get_evidence_clip",
        description=(
            "Locate the annotated video segment around a track and return its path, "
            "content hash and time bounds."
        ),
        args_model=GetEvidenceClipArgs,
        handler=get_evidence_clip,  # type: ignore[arg-type]
        effect=ToolEffect.READ_ONLY,
    ),
)
