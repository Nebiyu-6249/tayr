"""Security event logging.

Structured events for the things an incident responder needs: who authenticated, what
was refused, what was uploaded, and what the LLM was asked to do.

The hard rule here is that **no secret, token, or password ever reaches a log**. Logs
are copied to aggregators, shipped to third parties, and read by people who are not
supposed to have credentials. A token in a log is a credential in a place with weaker
access control than the database it came from.

Two mechanisms enforce that:

  - Structured fields only. There is no free-text interpolation of caller data into a
    message, so a caller cannot accidentally format a token into one.
  - `redact()` truncates any identifier to a short prefix, so events remain
    correlatable without being usable. Eight characters of a session token identify a
    session in a log while being useless for authentication.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

_logger = logging.getLogger("tayr.security")

# Enough to correlate two events, far too little to authenticate with.
_REDACT_KEEP = 8

# Field names whose values are never logged in full, whatever a caller passes.
_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "new_password",
        "current_password",
        "token",
        "session_token",
        "csrf_token",
        "password_hash",
        "token_hash",
        "secret",
        "api_key",
        "authorization",
        "cookie",
    }
)


class SecurityEvent(StrEnum):
    """Auditable events. Adding one here is how it becomes loggable."""

    LOGIN_SUCCEEDED = "login.succeeded"
    LOGIN_FAILED = "login.failed"
    LOGIN_LOCKED_OUT = "login.locked_out"
    LOGOUT = "logout"
    REGISTRATION = "registration"
    PASSWORD_CHANGED = "password.changed"  # noqa: S105 - an event name, not a credential
    SESSION_REJECTED = "session.rejected"
    AUTHZ_DENIED = "authz.denied"
    UPLOAD_ACCEPTED = "upload.accepted"
    UPLOAD_REJECTED = "upload.rejected"
    QUOTA_EXCEEDED = "quota.exceeded"
    RATE_LIMITED = "rate_limited"
    JOB_SUBMITTED = "job.submitted"
    JOB_FAILED = "job.failed"
    LLM_CALL = "llm.call"


def redact(value: str | None) -> str:
    """Reduce an identifier to a correlatable, unusable prefix."""
    if not value:
        return "-"
    if len(value) <= _REDACT_KEEP:
        return "*" * len(value)
    return f"{value[:_REDACT_KEEP]}..."


def _scrub(fields: dict[str, Any]) -> dict[str, Any]:
    """Redact anything whose key marks it sensitive, at any nesting level."""
    clean: dict[str, Any] = {}
    for key, value in fields.items():
        if key.lower() in _SENSITIVE_KEYS:
            clean[key] = redact(str(value)) if value is not None else "-"
        elif isinstance(value, dict):
            clean[key] = _scrub(value)
        else:
            clean[key] = value
    return clean


def log_security_event(
    event: SecurityEvent,
    *,
    user_id: str | None = None,
    client_ip: str | None = None,
    outcome: str = "ok",
    **fields: Any,
) -> dict[str, Any]:
    """Emit one structured security event. Returns the record, for testing.

    `user_id` and `client_ip` are recorded in full: they are the fields an incident
    responder needs to pivot on, and neither is a credential.
    """
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event.value,
        "outcome": outcome,
        "user_id": user_id or "-",
        "client_ip": client_ip or "-",
        **_scrub(fields),
    }
    _logger.info("security_event", extra={"security_event": record})
    return record
