"""The immutable decision record.

Every agent decision writes one of these, and a human must be able to reconstruct
exactly why the agent dismissed something from the record alone. That is both the safety
property and the demo: an operator who cannot audit a dismissal has no reason to trust
one, and dismissals are the product.

**Immutable by construction.** These are frozen dataclasses, and the database row is
never updated after the decision is written - operator feedback lands in a separate
table keyed to it. An audit record that can be edited after the fact is not an audit
record.

What is captured, and why each field is not optional:

  every tool call, with arguments AND results   reconstructing the reasoning needs both
  the computed verdict                          what actually happened
  the model's prose, separately                 so prose-vs-verdict divergence is visible
  prompt version and model name                 a decision made by a different prompt is
                                                a different decision
  token counts, wall-clock                      cost and latency attribution
  round count and whether the cap was hit       a capped loop is a distinct outcome
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from tayr.agent.verdicts import Attention, Uncertainty, Verdict

# Bumped whenever the system prompt or the rule set changes in a way that could change a
# decision. Recorded on every record so old decisions stay interpretable.
PROMPT_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """One tool call, as it happened."""

    tool_name: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0
    round_index: int = 0

    def redacted(self) -> dict[str, Any]:
        """Serialisable form. Tool arguments here are model-generated and already
        validated; they carry no secrets, but the result may carry a filesystem path,
        so paths are reduced to their basename."""
        result = self.result
        if result and "clip_path" in result:
            result = {**result, "clip_path": str(result["clip_path"]).rsplit("/", 1)[-1]}
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "ok": self.ok,
            "result": result,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "round_index": self.round_index,
        }


@dataclass(frozen=True, slots=True)
class VerdictDecision:
    """The computed verdict and the facts that produced it.

    `rule_id` names the single rule that fired. Every verdict traces to exactly one, so
    "why did it decide that" has a one-word answer before anyone reads any prose.
    """

    verdict: Verdict
    attention: Attention
    uncertainty: Uncertainty
    rule_id: str
    rationale: list[str] = field(default_factory=list)
    """Machine-generated factual bullets - measured values and thresholds. Not prose,
    and not written by a model."""


@dataclass(frozen=True, slots=True)
class AgentDecision:
    """The complete, immutable record of one triage decision."""

    track_id: str
    job_id: str
    site_id: str
    decision: VerdictDecision
    tool_calls: tuple[ToolInvocation, ...] = ()
    prose: str | None = None
    """The model's explanation. Never the source of the verdict - see `prose_diverged`."""

    prose_diverged: bool = False
    """True when the model's stated verdict disagreed with the computed one. The
    computed verdict wins; this flag records that it had to."""

    model: str = "none"
    prompt_version: str = PROMPT_VERSION
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rounds_used: int = 0
    round_cap_reached: bool = False
    duration_ms: float = 0.0
    synthetic: bool = False
    """True when any input came from a placeholder detector or classifier. Propagates to
    Slack, the API and the UI so a synthetic decision is never mistaken for a real one."""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "job_id": self.job_id,
            "site_id": self.site_id,
            "verdict": self.decision.verdict.value,
            "attention": self.decision.attention.value,
            "uncertainty": self.decision.uncertainty.value,
            "rule_id": self.decision.rule_id,
            "rationale": list(self.decision.rationale),
            "tool_calls": [t.redacted() for t in self.tool_calls],
            "prose": self.prose,
            "prose_diverged": self.prose_diverged,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "rounds_used": self.rounds_used,
            "round_cap_reached": self.round_cap_reached,
            "duration_ms": round(self.duration_ms, 2),
            "synthetic": self.synthetic,
            "created_at": self.created_at.isoformat(),
        }

    def audit_hash(self) -> str:
        """Content hash of the record, excluding wall-clock timings.

        Lets a stored record be checked for tampering, and lets two runs of the same
        decision be compared without duration noise. Timings vary between identical
        decisions; nothing else here should.
        """
        return audit_hash_of(self.to_dict())

    def as_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


#: Excluded from the audit hash: both vary between identical decisions and neither is
#: part of what was decided.
_UNHASHED = ("duration_ms", "created_at")


def audit_hash_of(record: dict[str, Any]) -> str:
    """The audit hash of a serialised decision.

    Takes the dict rather than the object so a record read back from `decisions.json`
    or from the database hashes identically to the one in memory - which is the whole
    point of a tamper check, and would not hold if a second caller reimplemented it.
    """
    payload = {k: v for k, v in record.items() if k not in _UNHASHED}
    payload["tool_calls"] = [
        {k: v for k, v in call.items() if k not in _UNHASHED}
        for call in record.get("tool_calls", [])
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
