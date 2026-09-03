"""The deterministic verdict rules. **This is the core of the system.**

The LLM writes prose. It does not decide anything. Every verdict is computed here, from
tool outputs, by rules a person can read and a test can pin. Three consequences follow,
and all three are the point:

1. **A prompt injection cannot change a verdict**, even if it survives every other
   layer, because no part of the model's output is consulted when deciding.
2. **The agent still works with the LLM switched off.** Verdicts happen; only the
   explanation is missing. `test_agent_loop.py` runs the whole thing with the provider
   forced to fail.
3. **Every decision has a `rule_id`**, so "why did it decide that" has a one-word answer
   before anyone reads a paragraph.

RULE ORDER MATTERS AND IS DELIBERATE. Rules are evaluated top to bottom and the first
match wins. Uncertainty is checked *before* any dismissal rule, so an unknown can never
be dismissed by a later rule that happened to match on partial data.

THRESHOLDS ARE `[ASSUMED]`, NOT MEASURED. The bird flapping band, the hover duration,
the small-target cut-off - none is fitted to data, because Tayr has no bird tracks to
fit against (docs/RESEARCH.md 14.4). They are design parameters chosen to be
conservative in the escalate direction, they are stated as assumptions wherever they
are quoted to an operator, and they must be validated before any claim rests on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tayr.agent.records import VerdictDecision
from tayr.agent.tools.readonly import (
    BIRD_FLAP_HZ_HIGH,
    BIRD_FLAP_HZ_LOW,
    MIN_HOVER_SECONDS,
    SMALL_TARGET_PX,
)
from tayr.agent.verdicts import Attention, Uncertainty, Verdict

# A dominant frequency needs this share of total spectral power before it counts as a
# real oscillation rather than the loudest bin of noise. `[ASSUMED]`
MIN_OSCILLATION_POWER = 0.35

# Below this, a track is watched rather than judged: too little history to say anything.
MIN_TRACK_SECONDS_TO_JUDGE = 2.0


@dataclass(frozen=True, slots=True)
class Evidence:
    """Tool outputs, gathered. The rules read only this.

    Assembled by the agent loop from `ToolInvocation` results. Keeping it separate from
    the loop means the rules can be tested exhaustively without a model, a network, or a
    database - which is why the rule tests are the fastest and most numerous in the
    suite.
    """

    analysis: dict[str, Any] | None = None
    authorization: dict[str, Any] | None = None
    zone: dict[str, Any] | None = None
    history: dict[str, Any] | None = None
    clip: dict[str, Any] | None = None
    failed_tools: tuple[str, ...] = ()
    round_cap_reached: bool = False


def _uncertain(
    reason: Uncertainty,
    rule_id: str,
    rationale: list[str],
    *,
    attention: Attention = Attention.PROMPT,
) -> VerdictDecision:
    """Escalate as uncertain.

    Never IMMEDIATE by default: "I could not tell" does not warrant the same urgency as
    a confident unexplained detection, and inflating it would train operators to ignore
    the loudest signal.
    """
    return VerdictDecision(
        verdict=Verdict.ESCALATE,
        attention=attention,
        uncertainty=reason,
        rule_id=rule_id,
        rationale=rationale,
    )


def decide(evidence: Evidence) -> VerdictDecision:
    """Compute the verdict. First matching rule wins.

    Returns ESCALATE for anything not positively explained. That asymmetry is the
    design: a missed page is worse than a noisy one.
    """
    rationale: list[str] = []

    # ---- Uncertainty first. An unknown must never fall through to a dismissal rule. ----

    if evidence.failed_tools:
        return _uncertain(
            Uncertainty.TOOL_FAILURE,
            "uncertain.tool_failure",
            [
                f"Tool(s) failed: {', '.join(evidence.failed_tools)}.",
                "The decision depended on evidence that did not arrive, so this is "
                "escalated as uncertain rather than judged on partial data.",
            ],
        )

    if evidence.round_cap_reached:
        return _uncertain(
            Uncertainty.ROUND_CAP_REACHED,
            "uncertain.round_cap",
            [
                "The reasoning loop reached its round cap without converging.",
                "Escalated as uncertain; a loop that does not finish is not a dismissal.",
            ],
        )

    if evidence.analysis is None:
        return _uncertain(
            Uncertainty.TOOL_FAILURE,
            "uncertain.no_analysis",
            ["No track analysis was gathered, so there is nothing to judge on."],
        )

    analysis = evidence.analysis
    duration = float(analysis.get("duration_seconds") or 0.0)
    pot = float(analysis.get("median_pixels_on_target") or 0.0)
    rationale.append(f"{pot:.1f} px on target over {duration:.1f}s.")

    # ---- Authorized flight: the largest source of dismissals, and confident. ----
    # Checked before the short-track rule: an authorized flight is explained regardless
    # of how much track history exists.

    auth = evidence.authorization
    if auth and auth.get("authorized"):
        return VerdictDecision(
            verdict=Verdict.DISMISS,
            attention=Attention.ROUTINE,
            uncertainty=Uncertainty.NONE,
            rule_id="dismiss.authorized_flight",
            rationale=[
                *rationale,
                f"Matches authorized flight {auth.get('flight_id')} ({auth.get('operator')}).",
                f"Match basis: {auth.get('match_basis')} - site and time only, not "
                "position or altitude.",
            ],
        )

    # ---- Too little history to judge. Watch, do not guess. ----

    if not analysis.get("features_available"):
        return VerdictDecision(
            verdict=Verdict.WATCH,
            attention=Attention.ROUTINE,
            uncertainty=Uncertainty.TRACK_TOO_SHORT,
            rule_id="watch.track_too_short",
            rationale=[
                *rationale,
                str(analysis.get("reason") or "No motion features available."),
                "Watching rather than judging: too little history to distinguish a bird "
                "from a multirotor.",
            ],
        )

    if duration < MIN_TRACK_SECONDS_TO_JUDGE:
        return VerdictDecision(
            verdict=Verdict.WATCH,
            attention=Attention.ROUTINE,
            uncertainty=Uncertainty.TRACK_TOO_SHORT,
            rule_id="watch.track_too_brief",
            rationale=[
                *rationale,
                f"Track is {duration:.1f}s, below the {MIN_TRACK_SECONDS_TO_JUDGE:.0f}s "
                "needed to characterise motion.",
            ],
        )

    features = analysis.get("features") or {}
    oscillation_hz = float(features.get("vertical_oscillation_hz") or 0.0)
    oscillation_power = float(features.get("vertical_oscillation_power") or 0.0)
    hover_seconds = float(analysis.get("hover_seconds") or 0.0)

    # ---- Sustained hover: birds do not station-keep. Escalate before the bird rule. ----

    if hover_seconds >= MIN_HOVER_SECONDS and oscillation_power < MIN_OSCILLATION_POWER:
        return VerdictDecision(
            verdict=Verdict.ESCALATE,
            attention=_attention_for_zone(evidence.zone),
            uncertainty=Uncertainty.NONE,
            rule_id="escalate.sustained_hover",
            rationale=[
                *rationale,
                f"Held position for {hover_seconds:.1f}s - birds do not hover.",
                f"Vertical oscillation {oscillation_hz:.1f} Hz at "
                f"{oscillation_power:.2f} power share, below the "
                f"{MIN_OSCILLATION_POWER} needed to count as flapping.",
                *_zone_rationale(evidence.zone),
                *_appearance_caveat(pot),
            ],
        )

    # ---- Flapping band: a real oscillation in the bird range. ----

    if (
        BIRD_FLAP_HZ_LOW <= oscillation_hz <= BIRD_FLAP_HZ_HIGH
        and oscillation_power >= MIN_OSCILLATION_POWER
    ):
        return VerdictDecision(
            verdict=Verdict.DISMISS,
            attention=Attention.ROUTINE,
            uncertainty=Uncertainty.NONE,
            rule_id="dismiss.flapping_band",
            rationale=[
                *rationale,
                f"Vertical oscillation {oscillation_hz:.1f} Hz at "
                f"{oscillation_power:.2f} power share, inside the "
                f"{BIRD_FLAP_HZ_LOW:.0f}-{BIRD_FLAP_HZ_HIGH:.0f} Hz flapping band.",
                f"Hover {hover_seconds:.1f}s - consistent with flapping flight.",
                "Band is an assumed design parameter, not fitted to measured bird tracks.",
            ],
        )

    # ---- Classifier, where one exists. ----

    classifier = analysis.get("classifier") or {}
    status = classifier.get("status")

    if status == "no_classifier_trained":
        return _uncertain(
            Uncertainty.NO_CLASSIFIER_TRAINED,
            "uncertain.no_classifier",
            [
                *rationale,
                "No trained track classifier in this deployment, so neither appearance "
                "nor motion classification is available.",
                "Motion did not match a hover or flapping pattern either, so nothing "
                "explains this track.",
                *_zone_rationale(evidence.zone),
                *_appearance_caveat(pot),
            ],
            attention=_attention_for_zone(evidence.zone, cap=Attention.PROMPT),
        )

    if status == "undetermined":
        interval = classifier.get("interval")
        return _uncertain(
            Uncertainty.CLASSIFIER_UNDETERMINED,
            "uncertain.classifier_undetermined",
            [
                *rationale,
                f"Classifier interval {interval} is too wide to call.",
                "Escalated as uncertain rather than dismissed on a point estimate.",
                *_zone_rationale(evidence.zone),
                *_appearance_caveat(pot),
            ],
            attention=_attention_for_zone(evidence.zone, cap=Attention.PROMPT),
        )

    if status == "determined" and classifier.get("label") in {"bird", "aircraft"}:
        return VerdictDecision(
            verdict=Verdict.DISMISS,
            attention=Attention.ROUTINE,
            uncertainty=Uncertainty.NONE,
            rule_id="dismiss.classifier_non_drone",
            rationale=[
                *rationale,
                f"Classifier: {classifier.get('label')} at "
                f"{classifier.get('confidence')} (interval {classifier.get('interval')}).",
            ],
        )

    # ---- Nothing explained it. ----

    return VerdictDecision(
        verdict=Verdict.ESCALATE,
        attention=_attention_for_zone(evidence.zone),
        uncertainty=Uncertainty.NONE,
        rule_id="escalate.unexplained",
        rationale=[
            *rationale,
            "No authorized flight, no flapping signature, no sustained hover, and no "
            "classifier verdict explains this track.",
            *_zone_rationale(evidence.zone),
            *_appearance_caveat(pot),
            *_history_rationale(evidence.history),
        ],
    )


def _attention_for_zone(zone: dict[str, Any] | None, *, cap: Attention | None = None) -> Attention:
    """How fast a human should look, from the airspace posture.

    Restricted airspace or an active restriction raises attention. **This orders a
    human's queue and nothing else** - see `verdicts.Attention`.
    """
    level = Attention.ROUTINE
    if zone:
        if zone.get("restriction_active") or zone.get("zone_class") == "restricted":
            level = Attention.IMMEDIATE
        elif zone.get("zone_class") == "controlled":
            level = Attention.PROMPT
        else:
            level = Attention.PROMPT
    else:
        level = Attention.PROMPT

    if cap is Attention.PROMPT and level is Attention.IMMEDIATE:
        # Uncertainty never reaches IMMEDIATE: "I could not tell" must not shout as
        # loudly as "I found something", or operators learn to ignore both.
        return Attention.PROMPT
    return level


def _zone_rationale(zone: dict[str, Any] | None) -> list[str]:
    if not zone:
        return []
    if not zone.get("site_known"):
        return ["Site is not in the registry, so no airspace posture is known for it."]
    parts = [f"Airspace: {zone.get('zone_class')}."]
    if zone.get("restriction_active"):
        parts.append(f"Temporary restriction active: {zone.get('restriction_reason')}.")
    return parts


def _appearance_caveat(pixels_on_target: float) -> list[str]:
    if pixels_on_target and pixels_on_target < SMALL_TARGET_PX:
        return [
            f"At {pixels_on_target:.0f} px on target, appearance classification is "
            f"unreliable (below {SMALL_TARGET_PX:.0f} px); this verdict is motion-based.",
        ]
    return []


def _history_rationale(history: dict[str, Any] | None) -> list[str]:
    if not history:
        return []
    count = int(history.get("match_count") or 0)
    if count == 0:
        return ["No similar track recorded at this site in the lookback window."]
    return [
        f"{count} similar track(s) at this site in the last "
        f"{history.get('window_days')} days - a repeat pattern, not a one-off."
    ]
