"""The two acting tools. **The only tools that change anything.**

There are exactly two, and both are idempotent per track. Everything else the agent can
reach is read-only, which is enforced by `ToolEffect` and asserted over the whole
registry in `test_agent_tools.py`.

Three controls sit between an injected instruction and a real page:

**Idempotency per track.** A second escalation for the same track updates the existing
message rather than posting again. An attacker who convinces the model to call
`escalate_to_human` fifty times produces one message, edited fifty times.

**Per-site rate limits.** A budget of acting-tool invocations per site per run, counted
in the tool context. Exhausting it fails the call rather than queueing it.

**A global circuit breaker.** Once tripped, no acting tool runs at all until it is
reset by hand. A runaway agent is a paging storm, and a paging storm is how a security
team learns to ignore the channel.

Note what is absent: nothing here deletes, modifies a model, changes configuration,
executes a command, or reaches the public internet.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from tayr.agent.tools.context import ToolContext
from tayr.agent.tools.readonly import TrackId, _Args, _require_track
from tayr.agent.tools.registry import ToolEffect, ToolSpec
from tayr.errors import TayrError
from tayr.security.audit import SecurityEvent, log_security_event

# Acting-tool invocations permitted per site in one run. Small: an agent triaging one
# track needs one escalation and one incident, not a stream.
DEFAULT_ACTION_BUDGET = 4

_BUDGET_KEY = "actions_remaining"
_BREAKER_KEY = "circuit_open"


class ActionRefusedError(TayrError):
    """An acting tool was refused by a budget or the circuit breaker."""


class EscalateArgs(_Args):
    """Note what the model does NOT supply: the verdict, the attention level, or the
    rationale. Those are computed and passed through from the decision record. The model
    can ask for an escalation to be delivered; it cannot author its content."""

    track_id: TrackId
    note: Annotated[str, Field(default="", max_length=500)]
    """Optional free text from the model. Rendered escaped, never interpreted."""


class OpenIncidentArgs(_Args):
    track_id: TrackId
    summary: Annotated[str, Field(default="", max_length=1000)]


def _spend(context: ToolContext, tool_name: str, track_id: str) -> None:
    """Consume one unit of the acting budget, or refuse."""
    if context.action_budget.get(_BREAKER_KEY):
        log_security_event(
            SecurityEvent.RATE_LIMITED, outcome="circuit_open", tool=tool_name, track_id=track_id
        )
        raise ActionRefusedError(
            "the acting-tool circuit breaker is open; no notifications will be sent "
            "until it is reset"
        )

    remaining = context.action_budget.get(_BUDGET_KEY, DEFAULT_ACTION_BUDGET)
    if remaining <= 0:
        context.action_budget[_BREAKER_KEY] = 1
        log_security_event(
            SecurityEvent.RATE_LIMITED,
            outcome="budget_exhausted",
            tool=tool_name,
            track_id=track_id,
            site_id=context.site_id,
        )
        raise ActionRefusedError(
            f"acting-tool budget exhausted for site {context.site_id}; circuit breaker tripped"
        )
    context.action_budget[_BUDGET_KEY] = remaining - 1


def escalate_to_human(context: ToolContext, args: EscalateArgs) -> dict[str, Any]:
    """Deliver an escalation to the on-call surface. Idempotent per track."""
    track = _require_track(context, args.track_id)
    _spend(context, "escalate_to_human", args.track_id)

    store = context.incident_store
    if store is None or context.notifier is None:
        raise TayrError("no notification surface is configured for this run")

    existing = store.notification_ref_for(args.track_id)  # type: ignore[attr-defined]
    decision = store.decision_for(args.track_id)  # type: ignore[attr-defined]
    if decision is None:
        raise TayrError(
            f"no decision record exists for track {args.track_id}; an escalation must "
            "carry the computed verdict, not the model's account of it"
        )

    notification = context.notifier.post_escalation(  # type: ignore[attr-defined]
        decision, existing_ref=existing
    )
    store.record_notification(args.track_id, notification.ref)  # type: ignore[attr-defined]

    log_security_event(
        SecurityEvent.JOB_SUBMITTED,
        outcome="escalated",
        track_id=args.track_id,
        site_id=context.site_id,
        updated=notification.updated,
        synthetic=context.synthetic,
    )
    return {
        "delivered": True,
        "channel": notification.channel,
        "ref": notification.ref,
        "updated": notification.updated,
        "track_number": track.track_number,
        "detail": (
            "updated the existing message for this track"
            if notification.updated
            else "posted a new escalation"
        ),
    }


def open_incident(context: ToolContext, args: OpenIncidentArgs) -> dict[str, Any]:
    """Create the incident record and start the audit trail. Idempotent per track."""
    _require_track(context, args.track_id)
    store = context.incident_store
    if store is None:
        raise TayrError("no incident store is configured for this run")

    existing = store.incident_for(args.track_id)  # type: ignore[attr-defined]
    if existing is not None:
        return {"incident_id": existing, "created": False, "detail": "incident already open"}

    _spend(context, "open_incident", args.track_id)
    incident_id = store.open_incident(args.track_id, args.summary)  # type: ignore[attr-defined]
    log_security_event(
        SecurityEvent.JOB_SUBMITTED,
        outcome="incident_opened",
        track_id=args.track_id,
        site_id=context.site_id,
    )
    return {"incident_id": incident_id, "created": True}


ACTING_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="escalate_to_human",
        description=(
            "Deliver the computed decision for a track to the on-call surface. "
            "Idempotent: calling again for the same track updates the existing message. "
            "You do not choose the verdict or write the message body."
        ),
        args_model=EscalateArgs,
        handler=escalate_to_human,  # type: ignore[arg-type]
        effect=ToolEffect.ACTING,
    ),
    ToolSpec(
        name="open_incident",
        description=(
            "Open an incident record for a track and start its audit trail. Idempotent per track."
        ),
        args_model=OpenIncidentArgs,
        handler=open_incident,  # type: ignore[arg-type]
        effect=ToolEffect.ACTING,
    ),
)
