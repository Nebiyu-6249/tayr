"""Slack adapter: signature verification, posting, and interactivity.

**Every inbound request is signature-verified before anything else happens.** An
unverified interactivity endpoint is an open API that mutates incident records: anyone
who finds the URL could confirm escalations, mark unauthorized flights as authorized,
and poison the feedback data the classifier will later be trained on.

The algorithm is implemented here rather than taken as a dependency, and it was read
from Slack's own SDK rather than written from memory
`[VERIFIED: slack_sdk 3.44.1, slack_sdk/signature/__init__.py]`:

    basestring  = f"v0:{timestamp}:{body}"
    signature   = "v0=" + HMAC_SHA256(signing_secret, basestring).hexdigest()
    compared with hmac.compare_digest against the X-Slack-Signature header
    X-Slack-Request-Timestamp rejected beyond +/- 300 seconds

The timestamp window is what stops a captured request being replayed later, and the
constant-time comparison is what stops the signature being recovered a byte at a time.

`[UNKNOWN]`: the interactivity payload arrives as a form-encoded `payload` field holding
JSON. Slack's documentation site is egress-blocked in this environment and the shape is
not described in their SDK, so `parse_interaction` is written to that assumption and is
the single function to correct if it is wrong. It validates what it finds rather than
trusting it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs

from tayr.agent.notify import Notification, render_blocks
from tayr.agent.records import AgentDecision
from tayr.errors import TayrError
from tayr.security.audit import SecurityEvent, log_security_event

SIGNATURE_HEADER = "x-slack-signature"
TIMESTAMP_HEADER = "x-slack-request-timestamp"
SIGNATURE_VERSION = "v0"
MAX_TIMESTAMP_SKEW_SECONDS = 300

VALID_ACTIONS = frozenset({"confirm", "dismiss_as_bird", "mark_authorized"})


class SlackVerificationError(TayrError):
    """An inbound request failed verification. Always a rejection, never a warning."""


def compute_signature(*, signing_secret: str, timestamp: str, body: str) -> str:
    """The expected `v0=...` signature for a request body."""
    basestring = f"{SIGNATURE_VERSION}:{timestamp}:{body}".encode()
    digest = hmac.new(signing_secret.encode("utf-8"), basestring, hashlib.sha256).hexdigest()
    return f"{SIGNATURE_VERSION}={digest}"


def verify_slack_request(
    *,
    signing_secret: str,
    body: str,
    headers: dict[str, str],
    now: float | None = None,
) -> None:
    """Verify an inbound Slack request, or raise.

    Raises rather than returning False: a caller that forgets to check a boolean gets a
    working endpoint with no verification, which is exactly the failure this prevents.
    """
    if not signing_secret:
        raise SlackVerificationError(
            "no Slack signing secret is configured; refusing to accept unverified requests"
        )

    lowered = {k.lower(): v for k, v in headers.items()}
    signature = lowered.get(SIGNATURE_HEADER)
    timestamp = lowered.get(TIMESTAMP_HEADER)

    if not signature or not timestamp:
        raise SlackVerificationError("missing Slack signature headers")

    try:
        sent_at = int(timestamp)
    except ValueError as exc:
        raise SlackVerificationError("malformed Slack timestamp") from exc

    current = now if now is not None else time.time()
    if abs(current - sent_at) > MAX_TIMESTAMP_SKEW_SECONDS:
        # Without this, a captured request replays forever.
        raise SlackVerificationError("Slack request timestamp is outside the accepted window")

    expected = compute_signature(signing_secret=signing_secret, timestamp=timestamp, body=body)
    if not hmac.compare_digest(expected, signature):
        log_security_event(SecurityEvent.AUTHZ_DENIED, outcome="slack_signature_invalid")
        raise SlackVerificationError("Slack signature verification failed")


@dataclass(frozen=True, slots=True)
class SlackInteraction:
    """A validated button press."""

    action_id: str
    track_id: str
    user_id: str
    user_name: str
    response_url: str | None = None
    message_ts: str | None = None


def parse_interaction(body: str) -> SlackInteraction:
    """Parse and validate an interactivity payload.

    Everything here is attacker-influenceable, so nothing is trusted: the action id must
    be one of the three known buttons, the track id is length-bounded, and the user name
    is treated as display text that will be escaped on output and never used for
    authorisation.
    """
    fields = parse_qs(body)
    raw = fields.get("payload", [None])[0]
    if not raw:
        raise SlackVerificationError("interactivity payload is missing")

    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SlackVerificationError("interactivity payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise SlackVerificationError("interactivity payload must be an object")

    actions = payload.get("actions") or []
    if not isinstance(actions, list) or not actions or not isinstance(actions[0], dict):
        raise SlackVerificationError("interactivity payload carries no action")

    action = actions[0]
    action_id = str(action.get("action_id") or "")
    if action_id not in VALID_ACTIONS:
        # An unknown action is not a request we understand. Refuse rather than guess.
        raise SlackVerificationError(f"unrecognised action: {action_id[:64]!r}")

    track_id = str(action.get("value") or "")[:64]
    if not track_id:
        raise SlackVerificationError("action carries no track id")

    user = payload.get("user") or {}
    return SlackInteraction(
        action_id=action_id,
        track_id=track_id,
        user_id=str(user.get("id") or "")[:64],
        user_name=str(user.get("name") or user.get("username") or "unknown")[:128],
        response_url=str(payload["response_url"])[:500] if payload.get("response_url") else None,
        message_ts=str((payload.get("message") or {}).get("ts") or "") or None,
    )


@dataclass
class SlackNotifier:
    """Posts decisions to Slack. Same message body as the local renderer.

    Idempotency lives in the caller: `existing_ref` is the message timestamp, and a
    second post for the same track becomes a `chat.update` rather than a `chat.post`.
    """

    bot_token: str
    alert_channel: str
    audit_channel: str
    client: Any = None
    """A slack_sdk WebClient, or any object exposing chat_postMessage / chat_update.
    Injected so the adapter can be tested without a workspace."""

    def _require_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.bot_token:
            raise TayrError("no Slack bot token is configured")
        try:
            from slack_sdk import WebClient
        except ImportError as exc:
            raise TayrError(
                "slack_sdk is not installed; install the 'slack' extra or use LocalNotifier"
            ) from exc
        self.client = WebClient(token=self.bot_token)
        return self.client

    def post_escalation(
        self, decision: AgentDecision, *, existing_ref: str | None = None
    ) -> Notification:
        client = self._require_client()
        blocks = render_blocks(decision)
        fallback = f"Tayr Watch: {decision.decision.verdict.value} on track {decision.track_id}"

        if existing_ref:
            client.chat_update(
                channel=self.alert_channel, ts=existing_ref, blocks=blocks, text=fallback
            )
            return Notification(ref=existing_ref, channel=self.alert_channel, updated=True)

        response = client.chat_postMessage(channel=self.alert_channel, blocks=blocks, text=fallback)
        return Notification(ref=str(response["ts"]), channel=self.alert_channel, updated=False)

    def post_dismissal(self, decision: AgentDecision) -> Notification:
        """Dismissals go to the audit channel. Never the alert channel."""
        client = self._require_client()
        response = client.chat_postMessage(
            channel=self.audit_channel,
            blocks=render_blocks(decision),
            text=f"Tayr Watch: dismissed track {decision.track_id}",
        )
        return Notification(ref=str(response["ts"]), channel=self.audit_channel, updated=False)
