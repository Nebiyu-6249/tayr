"""Agent decision endpoints, and the Slack interactivity callback.

Two authentication models live in this module and the difference matters.

**The read endpoints use the session cookie**, like every other Tayr route, and go
through `require_owned` so one tenant cannot read another's decisions.

**The Slack callback has no session at all.** Slack does not carry a Tayr cookie, so the
*request signature is the authentication*. It is verified before the body is parsed,
before anything is looked up, and before anything is written. If verification fails the
request is rejected outright - there is no partial path.

Because a Slack user is not a Tayr user, feedback is attributed to the decision's owner
and the Slack username is stored as display text only. It is never used to decide what
the request is allowed to touch.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from tayr.agent.slack import SlackVerificationError, parse_interaction, verify_slack_request
from tayr.api.auth import require_owned
from tayr.api.deps import CsrfProtected, CurrentUser, DbSession, Settings
from tayr.db.models import AgentDecisionRecord, Job, OperatorFeedbackRecord
from tayr.security.audit import SecurityEvent, log_security_event

router = APIRouter(tags=["agent"])

# Slack action id -> stored response. Anything else was already refused by
# parse_interaction; this mapping is the second place the set is pinned.
_RESPONSE_FOR_ACTION = {
    "confirm": "confirmed",
    "dismiss_as_bird": "dismissed_as_bird",
    "mark_authorized": "marked_authorized",
}


class DecisionOut(BaseModel):
    """Note what is absent: owner_id, and the raw record hash's inputs."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: str
    track_id: str
    job_id: str
    site_id: str
    verdict: str
    attention: str
    uncertainty: str
    rule_id: str
    prose_diverged: bool
    synthetic: bool
    model: str
    prompt_version: str
    rounds_used: int
    round_cap_reached: bool
    created_at: str
    audit_hash: str


class DecisionDetailOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: DecisionOut
    record: dict[str, object]
    """The full audit record: every tool call with arguments and results, the
    rationale, and the prose. This is what a human reconstructs the decision from."""

    feedback: list[dict[str, object]]


def _to_out(row: AgentDecisionRecord) -> DecisionOut:
    return DecisionOut(
        id=row.id,
        track_id=row.track_id,
        job_id=row.job_id,
        site_id=row.site_id,
        verdict=row.verdict,
        attention=row.attention,
        uncertainty=row.uncertainty,
        rule_id=row.rule_id,
        prose_diverged=row.prose_diverged,
        synthetic=row.synthetic,
        model=row.model,
        prompt_version=row.prompt_version,
        rounds_used=row.rounds_used,
        round_cap_reached=row.round_cap_reached,
        created_at=row.created_at.isoformat(),
        audit_hash=row.audit_hash,
    )


@router.get("/jobs/{job_id}/decisions", response_model=list[DecisionOut])
async def list_decisions(job_id: str, db: DbSession, user: CurrentUser) -> list[DecisionOut]:
    """Every agent decision for a job, oldest first."""
    job = await require_owned(db, Job, job_id, user)
    result = await db.execute(
        select(AgentDecisionRecord)
        .where(AgentDecisionRecord.job_id == job.id)
        .order_by(AgentDecisionRecord.created_at)
    )
    return [_to_out(row) for row in result.scalars().all()]


@router.get("/decisions/{decision_id}", response_model=DecisionDetailOut)
async def get_decision(decision_id: str, db: DbSession, user: CurrentUser) -> DecisionDetailOut:
    """One decision with its complete tool trace.

    The trace is the point: an operator who cannot audit a dismissal has no reason to
    trust one, and dismissals are what this system is for.
    """
    row = await require_owned(db, AgentDecisionRecord, decision_id, user)
    feedback_rows = await db.execute(
        select(OperatorFeedbackRecord)
        .where(OperatorFeedbackRecord.decision_id == row.id)
        .order_by(OperatorFeedbackRecord.created_at)
    )
    return DecisionDetailOut(
        decision=_to_out(row),
        record=json.loads(row.record_json),
        feedback=[
            {
                "response": f.response,
                "responder": f.responder,
                "note": f.note,
                "created_at": f.created_at.isoformat(),
            }
            for f in feedback_rows.scalars().all()
        ],
    )


class FeedbackIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response: Annotated[str, Field(pattern=r"^(confirmed|dismissed_as_bird|marked_authorized)$")]
    note: Annotated[str, Field(default="", max_length=1000)]


@router.post("/decisions/{decision_id}/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    decision_id: str,
    payload: FeedbackIn,
    db: DbSession,
    user: CurrentUser,
    _: CsrfProtected,
) -> dict[str, str]:
    """Record an operator's response from the web UI.

    Written alongside the decision, never over it. Agreement and disagreement are both
    signal, and overwriting would lose the disagreement.
    """
    row = await require_owned(db, AgentDecisionRecord, decision_id, user)
    db.add(
        OperatorFeedbackRecord(
            decision_id=row.id,
            owner_id=user.id,
            response=payload.response,
            responder=user.email,
            note=payload.note or None,
        )
    )
    log_security_event(
        SecurityEvent.JOB_SUBMITTED,
        user_id=user.id,
        outcome="feedback",
        decision_id=row.id,
        response=payload.response,
    )
    return {"status": "recorded"}


@router.post("/slack/interactions")
async def slack_interactions(request: Request, db: DbSession, settings: Settings) -> dict[str, str]:
    """Slack button callback.

    THE SIGNATURE IS THE AUTHENTICATION. There is no session here, so verification runs
    first, on the raw body, before parsing or any lookup. A failure is a rejection, not
    a warning.
    """
    raw_body = (await request.body()).decode("utf-8", errors="replace")

    try:
        verify_slack_request(
            signing_secret=settings.slack_signing_secret,
            body=raw_body,
            headers=dict(request.headers),
        )
    except SlackVerificationError as exc:
        log_security_event(
            SecurityEvent.AUTHZ_DENIED,
            outcome="slack_unverified",
            client_ip=request.client.host if request.client else None,
        )
        # 401 with no detail: an attacker probing this endpoint learns nothing about
        # why their forgery failed.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unauthorized") from exc

    try:
        interaction = parse_interaction(raw_body)
    except SlackVerificationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "malformed interaction") from exc

    result = await db.execute(
        select(AgentDecisionRecord).where(AgentDecisionRecord.track_id == interaction.track_id)
    )
    decision = result.scalar_one_or_none()
    if decision is None:
        # A verified request for a track we do not have. Not an error the presser can
        # fix, and not something to leak detail about.
        return {"text": "That track is no longer available."}

    db.add(
        OperatorFeedbackRecord(
            decision_id=decision.id,
            # A Slack user is not a Tayr user. Feedback belongs to the decision's owner;
            # the Slack name is display text and grants nothing.
            owner_id=decision.owner_id,
            response=_RESPONSE_FOR_ACTION[interaction.action_id],
            responder=interaction.user_name,
        )
    )
    log_security_event(
        SecurityEvent.JOB_SUBMITTED,
        outcome="slack_feedback",
        decision_id=decision.id,
        response=interaction.action_id,
    )
    return {"text": f"Recorded: {interaction.action_id.replace('_', ' ')}. Thank you."}
