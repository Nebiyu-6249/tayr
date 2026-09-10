"""The agent's decision space.

**This module defines the boundary of what Tayr Watch is allowed to decide, and the
enum below is exhaustive.** Three values, no fourth, and no sub-enum underneath any of
them that ranks anything.

The agent's authority ends at putting a decision in front of a person. It does not
recommend a response, does not order tracks by anything resembling engagement priority,
and does not interface with anything that acts on the physical world. Trajectory data is
used to answer "does this move like a bird"; it is never projected forward to say where
something will be. See CLAUDE.md section 2.

**ESCALATE is the safe default.** Every uncertainty - an UNDETERMINED classifier, a
failed tool call, a track too short to have features, a reasoning loop that hit its
round cap - resolves to ESCALATE with `Uncertainty` set, never to DISMISS. A missed page
is worse than a noisy one, and `test_agent_verdicts.py` enforces this rather than
leaving it to convention.
"""

from __future__ import annotations

from enum import StrEnum


class Verdict(StrEnum):
    """What the agent decided. Exhaustive - there is no fourth value."""

    DISMISS = "dismiss"
    """Explained: an authorized flight, or motion consistent with a bird or aircraft.
    Logged to the audit channel. No human is paged."""

    WATCH = "watch"
    """Not yet decidable, but not alarming either: too little track history to judge.
    Keeps tracking, posts a quiet thread update, re-evaluates later."""

    ESCALATE = "escalate"
    """A human should look. Pages the on-call with a full evidence package."""


class Attention(StrEnum):
    """How quickly a human should look at an escalation.

    **This is an attention level, not a threat ranking.** It answers "how soon should
    someone open this", derived from confidence and context. It is deliberately NOT a
    priority ordering for any response, and nothing downstream may treat it as one.
    Naming it `severity` invited that reading, so it is named for what it actually
    controls: the human's queue order, and nothing else.
    """

    ROUTINE = "routine"
    """Log it; someone reviews it in the normal course of a shift."""

    PROMPT = "prompt"
    """Worth interrupting for. The on-call is notified."""

    IMMEDIATE = "immediate"
    """Notify now. Reserved for a confident unexplained detection in a restricted
    context - not for uncertainty, which is PROMPT at most."""


class Uncertainty(StrEnum):
    """Why a decision could not be made confidently.

    Present on any ESCALATE that came from not knowing rather than from knowing. This
    is what lets an operator - and an auditor - tell "the agent found something" from
    "the agent could not tell", which are very different claims.
    """

    NONE = "none"
    CLASSIFIER_UNDETERMINED = "classifier_undetermined"
    """The track classifier's confidence interval was too wide to call."""

    TRACK_TOO_SHORT = "track_too_short"
    """Fewer observations than the motion features need to mean anything."""

    TOOL_FAILURE = "tool_failure"
    """A tool the decision depended on did not return."""

    ROUND_CAP_REACHED = "round_cap_reached"
    """The reasoning loop hit its cap without converging."""

    NO_CLASSIFIER_TRAINED = "no_classifier_trained"
    """There is no trained classifier in this deployment yet, so appearance and motion
    classification are both unavailable. Honest, and currently the common case."""

    CLASSIFIER_LOW_CONFIDENCE = "classifier_low_confidence"
    """The classifier named a class, with a tight enough interval to be called
    `determined`, but at a confidence too low to suppress a page on.

    A narrow interval says the model is consistent, not that it is right. Dismissing a
    track because a barely-better-than-chance classifier said "bird" is the failure this
    project is least willing to accept, so the confidence floor is separate from the
    interval width and both must pass."""


# Uncertainty reasons that must never produce a DISMISS. Used by the rule engine and
# asserted directly in tests, so the invariant lives in one place.
UNCERTAIN_REASONS: frozenset[Uncertainty] = frozenset(
    {
        Uncertainty.CLASSIFIER_UNDETERMINED,
        Uncertainty.TRACK_TOO_SHORT,
        Uncertainty.TOOL_FAILURE,
        Uncertainty.ROUND_CAP_REACHED,
        Uncertainty.NO_CLASSIFIER_TRAINED,
        Uncertainty.CLASSIFIER_LOW_CONFIDENCE,
    }
)
