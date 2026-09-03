"""Tayr Watch: the airspace triage agent.

Consumes finalised tracks from the pipeline, decides which deserve a human, and hands
escalations over as complete evidence packages.

The decision space is three verdicts and stops there - see `verdicts.py`, which is the
authoritative statement of what this agent is allowed to decide. The agent's authority
ends at putting a decision in front of a person.

Layering matches the rest of the codebase: this package is a library. The Slack handler
and the API routes are thin wrappers over it.
"""

from tayr.agent.records import (
    PROMPT_VERSION,
    AgentDecision,
    ToolInvocation,
    VerdictDecision,
)
from tayr.agent.verdicts import UNCERTAIN_REASONS, Attention, Uncertainty, Verdict

__all__ = [
    "PROMPT_VERSION",
    "UNCERTAIN_REASONS",
    "AgentDecision",
    "Attention",
    "ToolInvocation",
    "Uncertainty",
    "Verdict",
    "VerdictDecision",
]
