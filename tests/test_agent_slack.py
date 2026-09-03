"""Slack adapter tests.

Signature verification is the security boundary and gets the most attention. An
unverified interactivity endpoint is an open API that mutates incident records: anyone
who found the URL could confirm escalations, mark unauthorized flights authorized, and
poison the feedback data a classifier is later trained on.

The algorithm under test was read from Slack's own SDK
`[VERIFIED: slack_sdk 3.44.1, slack_sdk/signature/__init__.py]`, not from memory.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode

import pytest

from tayr.agent.slack import (
    MAX_TIMESTAMP_SKEW_SECONDS,
    VALID_ACTIONS,
    SlackInteraction,
    SlackNotifier,
    SlackVerificationError,
    compute_signature,
    parse_interaction,
    verify_slack_request,
)

SECRET = "8f742231b10e8888abcd99yyyzzz85a5"  # noqa: S105 - test fixture, not a credential
BODY = "payload=%7B%22type%22%3A%22block_actions%22%7D"


def headers_for(body: str, *, secret: str = SECRET, timestamp: str | None = None) -> dict[str, str]:
    ts = timestamp or str(int(time.time()))
    return {
        "X-Slack-Request-Timestamp": ts,
        "X-Slack-Signature": compute_signature(signing_secret=secret, timestamp=ts, body=body),
    }


def interaction_body(action_id: str = "confirm", track_id: str = "trk-1", user: str = "sam") -> str:
    payload = {
        "type": "block_actions",
        "user": {"id": "U123", "name": user},
        "response_url": "https://hooks.slack.example/actions/T/B/x",
        "message": {"ts": "1725360000.000100"},
        "actions": [{"action_id": action_id, "value": track_id, "type": "button"}],
    }
    return urlencode({"payload": json.dumps(payload)})


class TestSignatureAlgorithm:
    def test_matches_the_documented_basestring(self) -> None:
        """v0:{timestamp}:{body}, HMAC-SHA256, hex, prefixed v0=."""
        import hashlib
        import hmac

        ts = "1531420618"
        expected = (
            "v0="
            + hmac.new(SECRET.encode(), f"v0:{ts}:{BODY}".encode(), hashlib.sha256).hexdigest()
        )
        assert compute_signature(signing_secret=SECRET, timestamp=ts, body=BODY) == expected

    def test_signature_depends_on_the_body(self) -> None:
        ts = "1531420618"
        a = compute_signature(signing_secret=SECRET, timestamp=ts, body=BODY)
        b = compute_signature(signing_secret=SECRET, timestamp=ts, body=BODY + "x")
        assert a != b

    def test_signature_depends_on_the_timestamp(self) -> None:
        a = compute_signature(signing_secret=SECRET, timestamp="1", body=BODY)
        b = compute_signature(signing_secret=SECRET, timestamp="2", body=BODY)
        assert a != b


class TestVerification:
    def test_a_valid_request_passes(self) -> None:
        verify_slack_request(signing_secret=SECRET, body=BODY, headers=headers_for(BODY))

    def test_a_tampered_body_is_rejected(self) -> None:
        """The attack: capture a real request, change the track id, replay it."""
        headers = headers_for(BODY)
        with pytest.raises(SlackVerificationError, match="signature verification failed"):
            verify_slack_request(signing_secret=SECRET, body=BODY + "&evil=1", headers=headers)

    def test_a_wrong_secret_is_rejected(self) -> None:
        with pytest.raises(SlackVerificationError):
            verify_slack_request(
                signing_secret=SECRET,
                body=BODY,
                headers=headers_for(BODY, secret="wrong"),  # noqa: S106 - fixture
            )

    def test_a_forged_signature_is_rejected(self) -> None:
        with pytest.raises(SlackVerificationError):
            verify_slack_request(
                signing_secret=SECRET,
                body=BODY,
                headers={
                    "X-Slack-Request-Timestamp": str(int(time.time())),
                    "X-Slack-Signature": "v0=" + "0" * 64,
                },
            )

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"X-Slack-Signature": "v0=abc"},
            {"X-Slack-Request-Timestamp": "1725360000"},
        ],
    )
    def test_missing_headers_are_rejected(self, headers: dict[str, str]) -> None:
        with pytest.raises(SlackVerificationError, match="missing Slack signature headers"):
            verify_slack_request(signing_secret=SECRET, body=BODY, headers=headers)

    def test_an_old_request_is_rejected(self) -> None:
        """Without a timestamp window a captured request replays forever."""
        old = str(int(time.time()) - MAX_TIMESTAMP_SKEW_SECONDS - 60)
        with pytest.raises(SlackVerificationError, match="outside the accepted window"):
            verify_slack_request(
                signing_secret=SECRET, body=BODY, headers=headers_for(BODY, timestamp=old)
            )

    def test_a_future_request_is_rejected(self) -> None:
        future = str(int(time.time()) + MAX_TIMESTAMP_SKEW_SECONDS + 60)
        with pytest.raises(SlackVerificationError, match="outside the accepted window"):
            verify_slack_request(
                signing_secret=SECRET, body=BODY, headers=headers_for(BODY, timestamp=future)
            )

    def test_a_request_inside_the_window_passes(self) -> None:
        recent = str(int(time.time()) - MAX_TIMESTAMP_SKEW_SECONDS + 30)
        verify_slack_request(
            signing_secret=SECRET, body=BODY, headers=headers_for(BODY, timestamp=recent)
        )

    def test_a_malformed_timestamp_is_rejected(self) -> None:
        with pytest.raises(SlackVerificationError, match="malformed"):
            verify_slack_request(
                signing_secret=SECRET,
                body=BODY,
                headers={"X-Slack-Request-Timestamp": "not-a-number", "X-Slack-Signature": "v0=x"},
            )

    def test_no_configured_secret_refuses_rather_than_allows(self) -> None:
        """A missing secret must fail closed. Failing open here is an open API."""
        with pytest.raises(SlackVerificationError, match="refusing to accept unverified"):
            verify_slack_request(signing_secret="", body=BODY, headers=headers_for(BODY))

    def test_headers_are_matched_case_insensitively(self) -> None:
        ts = str(int(time.time()))
        verify_slack_request(
            signing_secret=SECRET,
            body=BODY,
            headers={
                "x-slack-request-timestamp": ts,
                "x-slack-signature": compute_signature(
                    signing_secret=SECRET, timestamp=ts, body=BODY
                ),
            },
        )

    def test_verification_raises_rather_than_returning_a_boolean(self) -> None:
        """A caller who forgets to check a boolean gets an endpoint with no
        verification. Raising removes that failure mode, so a change to a bool return
        must fail here."""
        import inspect

        assert inspect.signature(verify_slack_request).return_annotation in (None, "None")


class TestInteractionParsing:
    def test_parses_a_valid_button_press(self) -> None:
        result = parse_interaction(interaction_body("confirm", "trk-9", "alex"))
        assert isinstance(result, SlackInteraction)
        assert result.action_id == "confirm"
        assert result.track_id == "trk-9"
        assert result.user_name == "alex"

    @pytest.mark.parametrize("action", sorted(VALID_ACTIONS))
    def test_all_three_buttons_parse(self, action: str) -> None:
        assert parse_interaction(interaction_body(action)).action_id == action

    def test_an_unknown_action_is_refused_not_guessed(self) -> None:
        with pytest.raises(SlackVerificationError, match="unrecognised action"):
            parse_interaction(interaction_body("delete_everything"))

    def test_a_missing_payload_is_refused(self) -> None:
        with pytest.raises(SlackVerificationError, match="payload is missing"):
            parse_interaction("nothing=here")

    def test_malformed_json_is_refused(self) -> None:
        with pytest.raises(SlackVerificationError, match="not valid JSON"):
            parse_interaction(urlencode({"payload": "{not json"}))

    def test_a_payload_without_actions_is_refused(self) -> None:
        with pytest.raises(SlackVerificationError, match="carries no action"):
            parse_interaction(urlencode({"payload": json.dumps({"type": "block_actions"})}))

    def test_an_action_without_a_track_id_is_refused(self) -> None:
        payload = {"actions": [{"action_id": "confirm", "value": ""}], "user": {}}
        with pytest.raises(SlackVerificationError, match="no track id"):
            parse_interaction(urlencode({"payload": json.dumps(payload)}))

    def test_oversized_fields_are_truncated_not_rejected(self) -> None:
        """Display text from Slack is bounded rather than refused: a long username is
        not an attack, but an unbounded one reaching storage is a problem."""
        result = parse_interaction(interaction_body(user="x" * 5000))
        assert len(result.user_name) <= 128

    def test_user_name_is_display_text_only(self) -> None:
        """Never used for authorisation. Ownership comes from the decision record."""
        result = parse_interaction(interaction_body(user="<script>alert(1)</script>"))
        assert result.user_name == "<script>alert(1)</script>"  # escaped at render time


class FakeSlackClient:
    def __init__(self) -> None:
        self.posted: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []

    def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:  # noqa: N802 - Slack API name
        self.posted.append(kwargs)
        return {"ts": f"172536{len(self.posted):04d}.000100"}

    def chat_update(self, **kwargs: Any) -> dict[str, Any]:
        self.updated.append(kwargs)
        return {"ts": kwargs["ts"]}


class TestSlackNotifier:
    def _notifier(self) -> tuple[SlackNotifier, FakeSlackClient]:
        client = FakeSlackClient()
        return (
            SlackNotifier(
                bot_token="xoxb-test",  # noqa: S106 - test fixture
                alert_channel="#alerts",
                audit_channel="#audit",
                client=client,
            ),
            client,
        )

    def _decision(self) -> Any:
        from tayr.agent.llm import UnavailableLLM
        from tayr.agent.loop import TriageAgent
        from tests.test_agent_tools import context

        return TriageAgent(llm=UnavailableLLM()).triage("trk-1", context(), job_id="job-1")

    def test_first_post_creates_a_message(self) -> None:
        notifier, client = self._notifier()
        result = notifier.post_escalation(self._decision())
        assert result.updated is False
        assert len(client.posted) == 1
        assert client.posted[0]["channel"] == "#alerts"

    def test_second_post_updates_rather_than_reposting(self) -> None:
        notifier, client = self._notifier()
        decision = self._decision()
        first = notifier.post_escalation(decision)
        second = notifier.post_escalation(decision, existing_ref=first.ref)
        assert second.updated is True
        assert len(client.posted) == 1
        assert len(client.updated) == 1

    def test_dismissals_go_to_the_audit_channel(self) -> None:
        notifier, client = self._notifier()
        notifier.post_dismissal(self._decision())
        assert client.posted[0]["channel"] == "#audit"

    def test_a_fallback_text_is_always_sent(self) -> None:
        """Blocks do not render in notifications or on older clients."""
        notifier, client = self._notifier()
        notifier.post_escalation(self._decision())
        assert client.posted[0]["text"]
