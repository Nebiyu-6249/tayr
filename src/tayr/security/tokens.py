"""Session and CSRF tokens.

Every value here comes from `secrets`, never `random`. `random` is a Mersenne Twister:
observing a few hundred outputs lets an attacker reconstruct its internal state and
predict all future ones, which for session tokens means account takeover. Ruff rule
`S311` enforces this repository-wide.

Session tokens are stored **hashed**, like passwords. A database read - a SQL injection,
a leaked backup, an over-broad support query - then yields no usable session. SHA-256 is
correct here rather than Argon2: the token already has 256 bits of entropy, so there is
nothing to brute-force and no reason to pay a memory-hard cost on every request.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

# 32 bytes = 256 bits, url-safe base64 encoded.
_TOKEN_BYTES = 32


def generate_session_token() -> str:
    """A new session token. Give this to the client; store only `hash_token` of it."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def generate_csrf_token() -> str:
    """A new CSRF token."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def hash_token(token: str) -> str:
    """SHA-256 of a token, hex encoded. What gets persisted."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two secrets without leaking their common prefix length via timing.

    A plain `==` on strings short-circuits at the first differing byte. Over enough
    samples that timing difference lets an attacker recover a token one byte at a time.
    """
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
