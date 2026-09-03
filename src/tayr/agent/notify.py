"""The notification surface, behind an interface.

Slack is the target because security operations genuinely run on Slack and on-call
rotations. But the agent must not know that: `Notifier` is the whole contract, and the
local renderer is a first-class implementation rather than a mock. That is what lets the
demo run with no Slack workspace, and what would let a different surface be swapped in
without touching the agent.

`post_escalation` is **idempotent per track**. A second call for the same track updates
the existing message rather than posting again. Double-paging an on-call for one object
is how a system gets muted, and muting is the failure mode that matters most.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tayr.agent.records import AgentDecision
from tayr.errors import TayrError


class NotifyError(TayrError):
    """The notification surface could not be reached."""


@dataclass(frozen=True, slots=True)
class Notification:
    """What was posted, and where. `ref` is what makes an update possible."""

    ref: str
    channel: str
    updated: bool = False


@runtime_checkable
class Notifier(Protocol):
    """What the acting tools require of a notification surface."""

    def post_escalation(
        self, decision: AgentDecision, *, existing_ref: str | None = None
    ) -> Notification:
        """Post or update an escalation. Idempotent per track via `existing_ref`."""
        ...

    def post_dismissal(self, decision: AgentDecision) -> Notification:
        """Record a dismissal on the low-traffic audit channel.

        Never the alert channel. Suppression is the product and must be visible, but a
        dismissal that pages someone is not a suppression.
        """
        ...


def render_blocks(decision: AgentDecision) -> list[dict[str, Any]]:
    """Build the message body.

    Everything user-influenced is escaped. Operator names, filenames and notes are
    display text from an untrusted source, and a message body is a rendering context.
    """
    d = decision.decision
    header = {
        "escalate": ":rotating_light: Escalation",
        "watch": ":eyes: Watching",
        "dismiss": ":white_check_mark: Dismissed",
    }[d.verdict.value]

    lines = [f"*{header}* - track `{html.escape(decision.track_id)}`"]
    if decision.synthetic:
        lines.append(
            ":test_tube: *Synthetic run.* A placeholder detector produced this track. "
            "Nothing here describes real-world performance."
        )
    if d.uncertainty.value != "none":
        lines.append(f"_Uncertain: {d.uncertainty.value.replace('_', ' ')}_")

    lines.append("")
    lines.extend(f"• {html.escape(str(r))}" for r in d.rationale)

    if decision.prose:
        lines += ["", f"_{html.escape(decision.prose)}_"]
        if decision.prose_diverged:
            lines.append(
                ":warning: _The written explanation disagreed with the computed verdict. "
                "The computed verdict stands; this is logged as a defect._"
            )

    lines += [
        "",
        f"Rule `{d.rule_id}` · attention *{d.attention.value}* "
        f"(how soon a human should look, not a threat ranking)",
    ]

    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}}
    ]

    if d.verdict.value == "escalate":
        blocks.append(
            {
                "type": "actions",
                "block_id": f"tayr_decision:{decision.track_id}",
                "elements": [
                    {
                        "type": "button",
                        "action_id": "confirm",
                        "text": {"type": "plain_text", "text": "Confirm"},
                        "style": "primary",
                        "value": decision.track_id,
                    },
                    {
                        "type": "button",
                        "action_id": "dismiss_as_bird",
                        "text": {"type": "plain_text", "text": "Dismiss as bird"},
                        "value": decision.track_id,
                    },
                    {
                        "type": "button",
                        "action_id": "mark_authorized",
                        "text": {"type": "plain_text", "text": "Mark authorized"},
                        "value": decision.track_id,
                    },
                ],
            }
        )
    return blocks


@dataclass
class LocalNotifier:
    """Writes notifications to disk and renders an HTML view.

    The fallback implementation, and a real one: the demo runs on this when no Slack
    workspace is available, and it produces the identical message body so the two
    surfaces cannot drift.
    """

    output_dir: Path
    alert_channel: str = "#airspace-alerts"
    audit_channel: str = "#airspace-audit"
    posted: dict[str, dict[str, Any]] = field(default_factory=dict)

    def post_escalation(
        self, decision: AgentDecision, *, existing_ref: str | None = None
    ) -> Notification:
        ref = existing_ref or f"local-{decision.track_id}"
        updated = ref in self.posted
        self._write(ref, decision, self.alert_channel, updated)
        return Notification(ref=ref, channel=self.alert_channel, updated=updated)

    def post_dismissal(self, decision: AgentDecision) -> Notification:
        ref = f"local-dismiss-{decision.track_id}"
        updated = ref in self.posted
        self._write(ref, decision, self.audit_channel, updated)
        return Notification(ref=ref, channel=self.audit_channel, updated=updated)

    def _write(self, ref: str, decision: AgentDecision, channel: str, updated: bool) -> None:
        payload = {
            "ref": ref,
            "channel": channel,
            "updated": updated,
            "posted_at": datetime.now(UTC).isoformat(),
            "blocks": render_blocks(decision),
            "decision": decision.to_dict(),
        }
        self.posted[ref] = payload
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / f"{ref}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
