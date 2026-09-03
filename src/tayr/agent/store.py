"""Decision and incident storage.

An in-memory implementation plus the protocol the acting tools depend on. The database
implementation lives in the API layer; the agent depends only on this interface, so it
can be exercised without a database.

The important property: **the escalation carries the stored decision record**, not
anything the model said. `escalate_to_human` looks the decision up here rather than
accepting content from the model, so a message that reaches an on-call engineer always
shows the computed verdict and the machine-generated rationale.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tayr.agent.records import AgentDecision


@runtime_checkable
class DecisionStore(Protocol):
    """What acting tools require of storage."""

    def decision_for(self, track_id: str) -> AgentDecision | None: ...
    def notification_ref_for(self, track_id: str) -> str | None: ...
    def record_notification(self, track_id: str, ref: str) -> None: ...
    def incident_for(self, track_id: str) -> str | None: ...
    def open_incident(self, track_id: str, summary: str) -> str: ...


@dataclass
class InMemoryStore:
    """Storage for a single run, and for tests."""

    decisions: dict[str, AgentDecision] = field(default_factory=dict)
    notifications: dict[str, str] = field(default_factory=dict)
    incidents: dict[str, str] = field(default_factory=dict)
    incident_summaries: dict[str, str] = field(default_factory=dict)

    def put_decision(self, decision: AgentDecision) -> None:
        self.decisions[decision.track_id] = decision

    def decision_for(self, track_id: str) -> AgentDecision | None:
        return self.decisions.get(track_id)

    def notification_ref_for(self, track_id: str) -> str | None:
        return self.notifications.get(track_id)

    def record_notification(self, track_id: str, ref: str) -> None:
        self.notifications[track_id] = ref

    def incident_for(self, track_id: str) -> str | None:
        return self.incidents.get(track_id)

    def open_incident(self, track_id: str, summary: str) -> str:
        existing = self.incidents.get(track_id)
        if existing is not None:
            return existing
        incident_id = f"INC-{uuid.uuid4().hex[:10].upper()}"
        self.incidents[track_id] = incident_id
        self.incident_summaries[track_id] = summary
        return incident_id
