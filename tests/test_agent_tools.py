"""Agent tool tests.

The allowlist and argument-validation tests are the point. An agent with tools is a
privilege-escalation surface, and these check that the boundary holds against input the
model controls entirely.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tayr.agent.site_config import (
    AuthorizedFlight,
    SiteConfig,
    SiteRegistry,
    TemporaryRestriction,
)
from tayr.agent.tools import ToolEffect, UnknownToolError, build_registry
from tayr.agent.tools.context import HistoricalTrack, ToolContext, TrackSnapshot
from tayr.tracking.features import extract_features

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def features(kind: str = "hover", n: int = 60, fps: float = 30.0):  # type: ignore[no-untyped-def]
    t = np.arange(n, dtype=np.float64)
    if kind == "flapper":
        x, y = t * 3.0, 300.0 + 8.0 * np.sin(2 * np.pi * 5 * t / fps)
    else:
        x, y = np.full(n, 500.0), np.full(n, 300.0)
    side = 24.0
    boxes = np.stack([x - side / 2, y - side / 2, x + side / 2, y + side / 2], axis=-1)
    return extract_features(boxes, np.arange(n, dtype=np.int64), fps=fps)


def snapshot(track_id: str = "trk-1", *, feats: object = None, pot: float = 24.0) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=track_id,
        job_id="job-1",
        track_number=1,
        first_frame=0,
        last_frame=59,
        n_observations=60,
        median_pixels_on_target=pot,
        fps=30.0,
        features=feats,  # type: ignore[arg-type]
        observed_at=NOW,
    )


def context(**overrides: object) -> ToolContext:
    site = SiteConfig(
        site_id="site-a",
        name="Test Field",
        zone_class="restricted",
        authorized_flights=(
            AuthorizedFlight(
                flight_id="AUTH-1",
                operator="Survey Co",
                starts_at=NOW - timedelta(hours=1),
                ends_at=NOW + timedelta(hours=1),
            ),
        ),
        restrictions=(
            TemporaryRestriction(
                restriction_id="TR-1",
                reason="airshow",
                starts_at=NOW - timedelta(hours=2),
                ends_at=NOW + timedelta(hours=2),
            ),
        ),
    )
    base: dict[str, object] = {
        "site_id": "site-a",
        "tracks": {"trk-1": snapshot(feats=features())},
        "registry": SiteRegistry(sites=(site,)),
        "evidence_dir": Path("/nonexistent-evidence"),
        "now": NOW,
        "history": (),
    }
    base.update(overrides)
    return ToolContext(**base)  # type: ignore[arg-type]


class TestAllowlist:
    def test_registry_contains_only_the_declared_read_only_tools(self) -> None:
        assert build_registry().names == [
            "analyze_track",
            "check_airspace_zone",
            "check_authorization",
            "get_evidence_clip",
            "query_history",
        ]

    def test_read_only_registry_has_no_acting_tools(self) -> None:
        """A read-only run cannot post or open an incident because those tools are not
        in its registry at all, not because it chose not to call them."""
        assert build_registry().acting_tools() == set()

    def test_all_read_only_tools_declare_read_only_effect(self) -> None:
        assert all(s.effect is ToolEffect.READ_ONLY for s in build_registry().specs())

    @pytest.mark.parametrize(
        "name",
        ["run_shell", "delete_video", "eval", "analyze_track ", "ANALYZE_TRACK", "__import__"],
    )
    def test_unregistered_tool_name_is_a_hard_error(self, name: str) -> None:
        """Not a retry. A model naming a tool that does not exist gets an exception."""
        with pytest.raises(UnknownToolError, match="not a registered tool"):
            build_registry().invoke(name, {}, context())

    def test_unknown_tool_lists_what_is_registered_without_leaking_more(self) -> None:
        with pytest.raises(UnknownToolError) as exc:
            build_registry().invoke("escalate_to_human", {}, context())
        # Read-only registry: the acting tool is genuinely absent, not hidden.
        assert "escalate_to_human" not in str(exc.value).split("Registered:")[1]

    def test_openai_schema_shape_matches_the_sdk(self) -> None:
        """Shape verified against openai 3.3.1 chat_completion_function_tool_param.py."""
        schema = build_registry().specs()[0].openai_schema()
        assert schema["type"] == "function"
        assert set(schema["function"]) >= {"name", "description", "parameters"}
        assert schema["function"]["parameters"]["additionalProperties"] is False


class TestArgumentValidation:
    def test_valid_call_succeeds(self) -> None:
        call = build_registry().invoke("analyze_track", {"track_id": "trk-1"}, context())
        assert call.ok
        assert call.result is not None
        assert call.result["track_number"] == 1

    def test_unknown_argument_is_rejected(self) -> None:
        """extra='forbid': a hallucinated parameter fails rather than being ignored."""
        call = build_registry().invoke(
            "analyze_track", {"track_id": "trk-1", "override_verdict": "dismiss"}, context()
        )
        assert not call.ok
        assert "invalid arguments" in (call.error or "")

    def test_missing_required_argument_is_rejected(self) -> None:
        call = build_registry().invoke("analyze_track", {}, context())
        assert not call.ok

    def test_malformed_json_is_rejected(self) -> None:
        call = build_registry().invoke("analyze_track", "{not json", context())
        assert not call.ok
        assert "invalid arguments" in (call.error or "")

    def test_non_object_arguments_are_rejected(self) -> None:
        call = build_registry().invoke("analyze_track", '["trk-1"]', context())
        assert not call.ok

    @pytest.mark.parametrize(
        "bad_id",
        ["../../etc/passwd", "trk-1; DROP TABLE tracks", "trk 1", "a" * 200, "", "trk\x00"],
    )
    def test_track_id_is_character_bounded(self, bad_id: str) -> None:
        """A track_id is a uuid in practice. Anything that is not plausibly one is
        rejected before it reaches a lookup."""
        call = build_registry().invoke("analyze_track", {"track_id": bad_id}, context())
        assert not call.ok

    @pytest.mark.parametrize(
        ("args", "reason"),
        [
            ({"track_id": "trk-1", "pre_seconds": -1.0}, "negative"),
            ({"track_id": "trk-1", "pre_seconds": 10_000.0}, "unbounded"),
            ({"track_id": "trk-1", "post_seconds": 999.0}, "unbounded"),
        ],
    )
    def test_numeric_arguments_are_bounds_checked(
        self, args: dict[str, object], reason: str
    ) -> None:
        call = build_registry().invoke("get_evidence_clip", args, context())
        assert not call.ok, reason

    def test_track_from_another_job_is_refused(self) -> None:
        """The context holds only this job's tracks, so cross-job access fails by lookup
        rather than by a check someone could forget to write."""
        call = build_registry().invoke("analyze_track", {"track_id": "someone-elses"}, context())
        assert not call.ok
        assert "not part of this job" in (call.error or "")


class TestAnalyzeTrack:
    def test_reports_motion_features(self) -> None:
        result = build_registry().invoke("analyze_track", {"track_id": "trk-1"}, context()).result
        assert result is not None and result["features_available"] is True
        assert "vertical_oscillation_hz" in result["features"]
        assert result["bird_flap_band_hz"] == [2.0, 8.0]

    def test_absent_features_are_explained_not_zeroed(self) -> None:
        ctx = context(tracks={"trk-1": snapshot(feats=None)})
        result = build_registry().invoke("analyze_track", {"track_id": "trk-1"}, ctx).result
        assert result is not None and result["features_available"] is False
        assert "too few observations" in result["reason"]
        assert "features" not in result

    def test_untrained_classifier_is_distinguished_from_low_confidence(self) -> None:
        """'No classifier exists' and 'the classifier was unsure' are different claims
        and must not collapse into one."""
        result = build_registry().invoke("analyze_track", {"track_id": "trk-1"}, context()).result
        assert result is not None
        assert result["classifier"]["status"] == "no_classifier_trained"
        assert "absence of one" in result["classifier"]["detail"]

    def test_small_target_marks_appearance_unreliable(self) -> None:
        ctx = context(tracks={"trk-1": snapshot(feats=features(), pot=14.0)})
        result = build_registry().invoke("analyze_track", {"track_id": "trk-1"}, ctx).result
        assert result is not None and result["appearance_reliable"] is False


class TestAuthorization:
    def test_matching_flight_authorizes(self) -> None:
        result = (
            build_registry().invoke("check_authorization", {"track_id": "trk-1"}, context()).result
        )
        assert result is not None
        assert result["authorized"] is True
        assert result["flight_id"] == "AUTH-1"

    def test_match_basis_is_recorded_because_it_is_coarse(self) -> None:
        """Site and time only, not position or altitude. An auditor must see that."""
        result = (
            build_registry().invoke("check_authorization", {"track_id": "trk-1"}, context()).result
        )
        assert result is not None and result["match_basis"] == "site_and_time_window"

    def test_outside_the_window_is_not_authorized(self) -> None:
        late = replace(snapshot(feats=features()), observed_at=NOW + timedelta(days=2))
        ctx = context(tracks={"trk-1": late})
        result = build_registry().invoke("check_authorization", {"track_id": "trk-1"}, ctx).result
        assert result is not None and result["authorized"] is False

    def test_unknown_site_is_not_an_authorization(self) -> None:
        """An unknown site cannot vouch for anything, and reporting 'not authorized'
        would imply a registry that was actually consulted."""
        ctx = context(registry=SiteRegistry(sites=()))
        result = build_registry().invoke("check_authorization", {"track_id": "trk-1"}, ctx).result
        assert result is not None
        assert result["authorized"] is False
        assert result["site_known"] is False


class TestAirspaceZone:
    def test_reports_zone_and_active_restriction(self) -> None:
        result = (
            build_registry().invoke("check_airspace_zone", {"site_id": "site-a"}, context()).result
        )
        assert result is not None
        assert result["zone_class"] == "restricted"
        assert result["restriction_active"] is True
        assert result["restriction_reason"] == "airshow"

    def test_unknown_site_is_flagged_not_defaulted(self) -> None:
        result = (
            build_registry().invoke("check_airspace_zone", {"site_id": "nowhere"}, context()).result
        )
        assert result is not None and result["site_known"] is False


class TestQueryHistory:
    def test_finds_prior_tracks_in_the_size_band(self) -> None:
        ctx = context(
            history=(
                HistoricalTrack("old-1", NOW - timedelta(days=2), 24.0, "escalate"),
                HistoricalTrack("old-2", NOW - timedelta(days=400), 24.0, "escalate"),
                HistoricalTrack("old-3", NOW - timedelta(days=1), 200.0, "dismiss"),
            )
        )
        result = (
            build_registry()
            .invoke("query_history", {"size_band_px": 24.0, "window_days": 30}, ctx)
            .result
        )
        assert result is not None
        assert result["match_count"] == 1  # old-2 too old, old-3 wrong size
        assert result["matches"][0]["track_id"] == "old-1"

    def test_window_is_bounded(self) -> None:
        call = build_registry().invoke(
            "query_history", {"size_band_px": 24.0, "window_days": 100_000}, context()
        )
        assert not call.ok


class TestEvidenceClip:
    def test_path_is_constructed_not_supplied(self) -> None:
        """The model never supplies any part of the path, so no argument it invents can
        point somewhere else."""
        result = (
            build_registry().invoke("get_evidence_clip", {"track_id": "trk-1"}, context()).result
        )
        assert result is not None
        assert result["clip_path"].endswith("/trk-1.mp4")

    def test_missing_clip_is_reported_not_implied(self) -> None:
        result = (
            build_registry().invoke("get_evidence_clip", {"track_id": "trk-1"}, context()).result
        )
        assert result is not None
        assert result["exists"] is False
        assert result["sha256"] is None
        assert "no rendered clip" in result["detail"]

    def test_real_clip_is_hashed(self, tmp_path: Path) -> None:
        (tmp_path / "trk-1.mp4").write_bytes(b"fake clip bytes")
        result = (
            build_registry()
            .invoke("get_evidence_clip", {"track_id": "trk-1"}, context(evidence_dir=tmp_path))
            .result
        )
        assert result is not None
        assert result["exists"] is True
        assert len(result["sha256"]) == 64


class TestAuditTrail:
    def test_every_invocation_records_arguments_and_result(self) -> None:
        call = build_registry().invoke("analyze_track", {"track_id": "trk-1"}, context())
        record = call.redacted()
        assert record["tool_name"] == "analyze_track"
        assert record["arguments"] == {"track_id": "trk-1"}
        assert record["ok"] is True
        assert record["result"] is not None

    def test_failed_invocation_records_the_error(self) -> None:
        call = build_registry().invoke("analyze_track", {"track_id": "nope"}, context())
        assert call.redacted()["ok"] is False
        assert call.redacted()["error"]

    def test_clip_path_is_reduced_to_a_basename_in_the_record(self) -> None:
        """A full path in an audit record leaks server filesystem layout."""
        call = build_registry().invoke("get_evidence_clip", {"track_id": "trk-1"}, context())
        assert call.redacted()["result"]["clip_path"] == "trk-1.mp4"
