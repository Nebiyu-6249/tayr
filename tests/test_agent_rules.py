"""Verdict rule tests. **The most important tests in the agent.**

Verdicts are computed here, not by a model, so these tests are the specification of what
Tayr Watch decides. They run without a model, a network or a database, which is why they
are exhaustive.

The invariant that matters most is at the bottom: no uncertainty of any kind can produce
a DISMISS. It is asserted over the whole `Uncertainty` enum rather than case by case, so
adding a new uncertainty reason without handling it fails the suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from tayr.agent.rules import (
    MIN_OSCILLATION_POWER,
    MIN_TRACK_SECONDS_TO_JUDGE,
    Evidence,
    decide,
)
from tayr.agent.verdicts import UNCERTAIN_REASONS, Attention, Uncertainty, Verdict


def analysis(
    *,
    duration: float = 10.0,
    pot: float = 24.0,
    features: bool = True,
    oscillation_hz: float = 0.0,
    oscillation_power: float = 0.0,
    hover_seconds: float = 0.0,
    classifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "duration_seconds": duration,
        "median_pixels_on_target": pot,
        "appearance_reliable": pot >= 20.0,
        "features_available": features,
        "classifier": classifier or {"status": "no_classifier_trained", "detail": "none"},
    }
    if features:
        payload["features"] = {
            "vertical_oscillation_hz": oscillation_hz,
            "vertical_oscillation_power": oscillation_power,
            "mean_speed_px_per_frame": 2.0,
        }
        payload["hover_seconds"] = hover_seconds
    else:
        payload["reason"] = "too few observations for motion features"
    return payload


ZONE_RESTRICTED = {
    "site_known": True,
    "zone_class": "restricted",
    "restriction_active": True,
    "restriction_reason": "airshow",
}
ZONE_OPEN = {"site_known": True, "zone_class": "unrestricted", "restriction_active": False}
AUTHORIZED = {
    "authorized": True,
    "flight_id": "AUTH-1",
    "operator": "Survey Co",
    "match_basis": "site_and_time_window",
}
NOT_AUTHORIZED = {"authorized": False, "match_basis": "site_and_time_window"}


class TestRequiredCases:
    """The six cases the build order names explicitly."""

    def test_hover_with_no_oscillation_escalates(self) -> None:
        """Birds do not station-keep. This is the signature that matters most."""
        d = decide(
            Evidence(
                analysis=analysis(hover_seconds=11.2, oscillation_hz=0.3, oscillation_power=0.05),
                authorization=NOT_AUTHORIZED,
                zone=ZONE_RESTRICTED,
            )
        )
        assert d.verdict is Verdict.ESCALATE
        assert d.rule_id == "escalate.sustained_hover"
        assert d.uncertainty is Uncertainty.NONE
        assert any("do not hover" in r for r in d.rationale)

    def test_flapping_band_dismisses_as_bird(self) -> None:
        d = decide(
            Evidence(
                analysis=analysis(oscillation_hz=5.0, oscillation_power=0.66),
                authorization=NOT_AUTHORIZED,
                zone=ZONE_OPEN,
            )
        )
        assert d.verdict is Verdict.DISMISS
        assert d.rule_id == "dismiss.flapping_band"

    def test_authorized_flight_dismisses(self) -> None:
        d = decide(
            Evidence(
                analysis=analysis(hover_seconds=30.0),
                authorization=AUTHORIZED,
                zone=ZONE_RESTRICTED,
            )
        )
        assert d.verdict is Verdict.DISMISS
        assert d.rule_id == "dismiss.authorized_flight"
        assert any("AUTH-1" in r for r in d.rationale)

    def test_undetermined_classifier_escalates_as_uncertain(self) -> None:
        d = decide(
            Evidence(
                analysis=analysis(
                    classifier={
                        "status": "undetermined",
                        "interval": [0.51, 0.79],
                        "label": "drone",
                    }
                ),
                authorization=NOT_AUTHORIZED,
                zone=ZONE_OPEN,
            )
        )
        assert d.verdict is Verdict.ESCALATE
        assert d.uncertainty is Uncertainty.CLASSIFIER_UNDETERMINED
        assert any("too wide to call" in r for r in d.rationale)

    def test_short_track_is_watched_not_judged(self) -> None:
        d = decide(Evidence(analysis=analysis(features=False), authorization=NOT_AUTHORIZED))
        assert d.verdict is Verdict.WATCH
        assert d.rule_id == "watch.track_too_short"
        assert d.uncertainty is Uncertainty.TRACK_TOO_SHORT

    def test_tool_failure_escalates_as_uncertain(self) -> None:
        d = decide(Evidence(analysis=analysis(), failed_tools=("check_authorization",)))
        assert d.verdict is Verdict.ESCALATE
        assert d.uncertainty is Uncertainty.TOOL_FAILURE
        assert any("check_authorization" in r for r in d.rationale)


class TestRuleOrdering:
    """Order is deliberate. These pin it so a reorder is a test failure, not a surprise."""

    def test_tool_failure_beats_everything_including_a_dismissal(self) -> None:
        """An unknown must never fall through to a rule that happened to match on
        partial data."""
        d = decide(
            Evidence(
                analysis=analysis(oscillation_hz=5.0, oscillation_power=0.9),
                authorization=AUTHORIZED,
                failed_tools=("analyze_track",),
            )
        )
        assert d.verdict is Verdict.ESCALATE
        assert d.uncertainty is Uncertainty.TOOL_FAILURE

    def test_authorization_beats_a_short_track(self) -> None:
        """An authorized flight is explained regardless of how much history exists."""
        d = decide(Evidence(analysis=analysis(features=False), authorization=AUTHORIZED))
        assert d.verdict is Verdict.DISMISS

    def test_hover_beats_the_flapping_band(self) -> None:
        """Something both hovering and oscillating weakly is not a bird."""
        d = decide(
            Evidence(
                analysis=analysis(hover_seconds=8.0, oscillation_hz=5.0, oscillation_power=0.1),
                authorization=NOT_AUTHORIZED,
            )
        )
        assert d.verdict is Verdict.ESCALATE
        assert d.rule_id == "escalate.sustained_hover"

    def test_round_cap_escalates_before_any_dismissal(self) -> None:
        d = decide(
            Evidence(
                analysis=analysis(oscillation_hz=5.0, oscillation_power=0.9),
                authorization=AUTHORIZED,
                round_cap_reached=True,
            )
        )
        assert d.verdict is Verdict.ESCALATE
        assert d.uncertainty is Uncertainty.ROUND_CAP_REACHED


class TestOscillationBand:
    @pytest.mark.parametrize("hz", [2.0, 5.0, 8.0])
    def test_inside_the_band_dismisses(self, hz: float) -> None:
        d = decide(
            Evidence(
                analysis=analysis(oscillation_hz=hz, oscillation_power=0.7),
                authorization=NOT_AUTHORIZED,
            )
        )
        assert d.verdict is Verdict.DISMISS

    @pytest.mark.parametrize("hz", [0.3, 1.9, 8.1, 15.0])
    def test_outside_the_band_does_not_dismiss(self, hz: float) -> None:
        d = decide(
            Evidence(
                analysis=analysis(oscillation_hz=hz, oscillation_power=0.7),
                authorization=NOT_AUTHORIZED,
            )
        )
        assert d.verdict is not Verdict.DISMISS

    def test_a_frequency_without_power_is_noise_not_flapping(self) -> None:
        """A dominant bin with negligible power is the loudest noise, not a wingbeat."""
        d = decide(
            Evidence(
                analysis=analysis(
                    oscillation_hz=5.0, oscillation_power=MIN_OSCILLATION_POWER - 0.01
                ),
                authorization=NOT_AUTHORIZED,
            )
        )
        assert d.verdict is not Verdict.DISMISS

    def test_brief_track_is_watched_even_with_features(self) -> None:
        d = decide(
            Evidence(
                analysis=analysis(duration=MIN_TRACK_SECONDS_TO_JUDGE - 0.1),
                authorization=NOT_AUTHORIZED,
            )
        )
        assert d.verdict is Verdict.WATCH
        assert d.rule_id == "watch.track_too_brief"


class TestAttention:
    """Attention orders a human's queue. It is not a threat ranking."""

    def test_restricted_airspace_raises_attention(self) -> None:
        d = decide(
            Evidence(
                analysis=analysis(hover_seconds=10.0),
                authorization=NOT_AUTHORIZED,
                zone=ZONE_RESTRICTED,
            )
        )
        assert d.attention is Attention.IMMEDIATE

    def test_open_airspace_does_not(self) -> None:
        d = decide(
            Evidence(
                analysis=analysis(hover_seconds=10.0),
                authorization=NOT_AUTHORIZED,
                zone=ZONE_OPEN,
            )
        )
        assert d.attention is not Attention.IMMEDIATE

    def test_uncertainty_never_reaches_immediate(self) -> None:
        """'I could not tell' must not shout as loudly as 'I found something', or
        operators learn to ignore both."""
        d = decide(
            Evidence(
                analysis=analysis(
                    classifier={"status": "undetermined", "interval": [0.4, 0.9], "label": "drone"}
                ),
                authorization=NOT_AUTHORIZED,
                zone=ZONE_RESTRICTED,
            )
        )
        assert d.verdict is Verdict.ESCALATE
        assert d.attention is not Attention.IMMEDIATE

    def test_dismissals_are_always_routine(self) -> None:
        d = decide(Evidence(analysis=analysis(), authorization=AUTHORIZED, zone=ZONE_RESTRICTED))
        assert d.attention is Attention.ROUTINE


class TestRationaleIsPhysical:
    """Reasoning must be checkable against the video, not a confidence number."""

    def test_rationale_quotes_measured_values(self) -> None:
        d = decide(
            Evidence(
                analysis=analysis(
                    hover_seconds=11.2, oscillation_hz=0.3, oscillation_power=0.05, pot=14.0
                ),
                authorization=NOT_AUTHORIZED,
                zone=ZONE_RESTRICTED,
            )
        )
        joined = " ".join(d.rationale)
        assert "11.2s" in joined
        assert "0.3 Hz" in joined
        assert "14" in joined

    def test_small_targets_carry_the_appearance_caveat(self) -> None:
        d = decide(
            Evidence(analysis=analysis(hover_seconds=10.0, pot=14.0), authorization=NOT_AUTHORIZED)
        )
        assert any("appearance classification is unreliable" in r for r in d.rationale)

    def test_large_targets_do_not(self) -> None:
        d = decide(
            Evidence(analysis=analysis(hover_seconds=10.0, pot=80.0), authorization=NOT_AUTHORIZED)
        )
        assert not any("appearance classification is unreliable" in r for r in d.rationale)

    def test_assumed_thresholds_are_labelled_to_the_operator(self) -> None:
        """The flapping band is not fitted to data, and the operator is told so."""
        d = decide(
            Evidence(
                analysis=analysis(oscillation_hz=5.0, oscillation_power=0.7),
                authorization=NOT_AUTHORIZED,
            )
        )
        assert any("assumed design parameter" in r for r in d.rationale)

    def test_authorization_match_basis_is_disclosed(self) -> None:
        d = decide(Evidence(analysis=analysis(), authorization=AUTHORIZED))
        assert any("site and time only" in r for r in d.rationale)

    def test_every_verdict_carries_a_rule_id_and_rationale(self) -> None:
        for evidence in (
            Evidence(analysis=analysis(), authorization=AUTHORIZED),
            Evidence(analysis=analysis(features=False)),
            Evidence(analysis=analysis(hover_seconds=9.0)),
            Evidence(failed_tools=("x",)),
            Evidence(),
        ):
            d = decide(evidence)
            assert d.rule_id
            assert d.rationale


class TestEscalateOnUncertaintyInvariant:
    """The safety property, asserted over the whole enum rather than case by case."""

    @pytest.mark.parametrize("reason", sorted(UNCERTAIN_REASONS, key=str))
    def test_no_uncertainty_reason_can_produce_a_dismissal(self, reason: Uncertainty) -> None:
        """Adding a new uncertainty reason without handling it fails here."""
        builders = {
            Uncertainty.TOOL_FAILURE: Evidence(analysis=analysis(), failed_tools=("t",)),
            Uncertainty.ROUND_CAP_REACHED: Evidence(analysis=analysis(), round_cap_reached=True),
            Uncertainty.TRACK_TOO_SHORT: Evidence(analysis=analysis(features=False)),
            Uncertainty.CLASSIFIER_UNDETERMINED: Evidence(
                analysis=analysis(classifier={"status": "undetermined", "interval": [0.4, 0.9]}),
            ),
            Uncertainty.NO_CLASSIFIER_TRAINED: Evidence(analysis=analysis()),
        }
        evidence = builders[reason]
        d = decide(evidence)
        assert d.verdict is not Verdict.DISMISS, f"{reason} produced a dismissal"
        assert d.uncertainty is reason

    def test_empty_evidence_escalates(self) -> None:
        """The degenerate case: nothing gathered at all must not be a dismissal."""
        d = decide(Evidence())
        assert d.verdict is Verdict.ESCALATE

    def test_unexplained_track_escalates(self) -> None:
        d = decide(
            Evidence(
                analysis=analysis(
                    oscillation_hz=12.0,
                    oscillation_power=0.8,
                    classifier={
                        "status": "determined",
                        "label": "drone",
                        "confidence": 0.9,
                        "interval": [0.85, 0.95],
                    },
                ),
                authorization=NOT_AUTHORIZED,
                zone=ZONE_RESTRICTED,
            )
        )
        assert d.verdict is Verdict.ESCALATE
        assert d.rule_id == "escalate.unexplained"

    def test_only_three_verdicts_are_reachable(self) -> None:
        """The decision space is exhaustive. Fuzz the inputs; nothing else appears."""
        seen = set()
        for hover in (0.0, 5.0, 20.0):
            for hz in (0.0, 3.0, 12.0):
                for power in (0.0, 0.5, 0.99):
                    for auth in (AUTHORIZED, NOT_AUTHORIZED, None):
                        for feats in (True, False):
                            d = decide(
                                Evidence(
                                    analysis=analysis(
                                        features=feats,
                                        oscillation_hz=hz,
                                        oscillation_power=power,
                                        hover_seconds=hover,
                                    ),
                                    authorization=auth,
                                    zone=ZONE_OPEN,
                                )
                            )
                            seen.add(d.verdict)
        assert seen <= {Verdict.DISMISS, Verdict.WATCH, Verdict.ESCALATE}
