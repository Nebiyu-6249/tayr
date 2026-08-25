"""Security event logging tests.

The point of these is one rule: no credential reaches a log. Logs travel further and
are read by more people than the database they describe.
"""

from __future__ import annotations

import pytest

from tayr.security.audit import SecurityEvent, log_security_event, redact


class TestRedaction:
    def test_long_value_keeps_only_a_prefix(self) -> None:
        token = "abcdefghijklmnopqrstuvwxyz0123456789"  # noqa: S105 - fixture
        out = redact(token)
        assert out == "abcdefgh..."
        assert token not in out

    def test_short_value_is_fully_masked(self) -> None:
        """A short secret has too little prefix to be safe to show at all."""
        assert redact("secret") == "******"

    def test_none_and_empty(self) -> None:
        assert redact(None) == "-"
        assert redact("") == "-"


class TestEventScrubbing:
    @pytest.mark.parametrize(
        "key",
        ["password", "new_password", "token", "csrf_token", "password_hash", "api_key", "cookie"],
    )
    def test_sensitive_fields_never_logged_in_full(self, key: str) -> None:
        secret = "super-secret-value-that-must-not-appear-anywhere"  # noqa: S105
        record = log_security_event(SecurityEvent.LOGIN_FAILED, **{key: secret})
        assert secret not in str(record)

    def test_nested_sensitive_fields_are_scrubbed(self) -> None:
        secret = "another-secret-value-entirely-here"  # noqa: S105
        record = log_security_event(
            SecurityEvent.LLM_CALL, context={"api_key": secret, "model": "gpt-5-mini"}
        )
        assert secret not in str(record)
        assert record["context"]["model"] == "gpt-5-mini"

    def test_non_sensitive_fields_survive(self) -> None:
        record = log_security_event(SecurityEvent.UPLOAD_ACCEPTED, container="mp4", size_bytes=1234)
        assert record["container"] == "mp4"
        assert record["size_bytes"] == 1234

    def test_user_and_ip_are_kept_for_incident_response(self) -> None:
        record = log_security_event(
            SecurityEvent.LOGIN_SUCCEEDED, user_id="user-123", client_ip="203.0.113.7"
        )
        assert record["user_id"] == "user-123"
        assert record["client_ip"] == "203.0.113.7"

    def test_record_shape(self) -> None:
        record = log_security_event(SecurityEvent.LOGOUT)
        assert set(record) >= {"timestamp", "event", "outcome", "user_id", "client_ip"}
        assert record["event"] == "logout"
