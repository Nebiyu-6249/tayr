"""Rate limiting and progressive login lockout.

Two related but distinct controls:

**Rate limiting** caps how often an action may be attempted, protecting expensive
endpoints (upload, password reset, LLM calls) from abuse. A token bucket is used rather
than a fixed window, because fixed windows allow a burst of 2x the limit across a window
boundary.

**Progressive lockout** slows password guessing specifically. Delay grows with
consecutive failures, so an online attack becomes impractical while a person who
mistypes twice barely notices. Lockout is keyed on the **account**, not the client
address: keying on address alone lets an attacker with many addresses guess freely, and
locking an address can also shut out everyone behind one NAT.

Both are implemented against an injected clock so their behaviour over time can be
tested without sleeping.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from tayr.errors import TayrError


class RateLimitedError(TayrError):
    """An action was attempted too often."""

    def __init__(self, message: str, *, retry_after_seconds: float) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


@dataclass
class TokenBucket:
    """Allows `capacity` actions, refilling at `refill_per_second`."""

    capacity: float
    refill_per_second: float
    tokens: float = field(init=False)
    _last: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.refill_per_second <= 0:
            raise ValueError("capacity and refill_per_second must be positive")
        self.tokens = self.capacity

    def consume(self, now: float, amount: float = 1.0) -> None:
        """Take `amount` tokens, or raise RateLimitedError with a retry hint."""
        elapsed = max(0.0, now - self._last)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self._last = now

        if self.tokens < amount:
            shortfall = amount - self.tokens
            raise RateLimitedError(
                "too many requests",
                retry_after_seconds=shortfall / self.refill_per_second,
            )
        self.tokens -= amount


class RateLimiter:
    """Per-key token buckets. Backed by a dict here; Redis in a multi-process deployment.

    NOTE: this in-process implementation is correct for a single API worker and for
    tests. With several workers each holds its own buckets, so the effective limit
    multiplies by the worker count. A deployment running more than one worker must move
    this to Redis - recorded as an accepted risk in docs/THREAT_MODEL.md rather than
    left as a surprise.
    """

    def __init__(
        self,
        *,
        capacity: float,
        refill_per_second: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._capacity = capacity
        self._refill = refill_per_second
        self._buckets: dict[str, TokenBucket] = {}
        self._clock = clock or _default_clock

    def check(self, key: str, amount: float = 1.0) -> None:
        """Consume from `key`'s bucket, raising RateLimitedError if exhausted."""
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(capacity=self._capacity, refill_per_second=self._refill)
            bucket._last = self._clock()
            self._buckets[key] = bucket
        bucket.consume(self._clock(), amount)

    def reset(self, key: str) -> None:
        self._buckets.pop(key, None)


def _default_clock() -> float:
    import time

    return time.monotonic()


# Delay after N consecutive failures, in seconds. Doubling, capped.
_LOCKOUT_BASE_SECONDS = 1.0
_LOCKOUT_CAP_SECONDS = 900.0
_FREE_ATTEMPTS = 3
"""Attempts before any delay applies. A person who mistypes twice should not be
punished; an attacker running thousands of guesses should be."""


def lockout_delay(consecutive_failures: int) -> float:
    """Seconds a login attempt must wait, given consecutive prior failures."""
    if consecutive_failures <= _FREE_ATTEMPTS:
        return 0.0
    exponent = consecutive_failures - _FREE_ATTEMPTS - 1
    return min(_LOCKOUT_BASE_SECONDS * math.pow(2, exponent), _LOCKOUT_CAP_SECONDS)


@dataclass
class LoginAttemptTracker:
    """Tracks consecutive failures per account for progressive lockout."""

    _failures: dict[str, int] = field(default_factory=dict)
    _locked_until: dict[str, float] = field(default_factory=dict)

    def check(self, account_key: str, now: float) -> None:
        """Raise if this account is currently in a lockout window."""
        until = self._locked_until.get(account_key)
        if until is not None and now < until:
            raise RateLimitedError("too many failed attempts", retry_after_seconds=until - now)

    def record_failure(self, account_key: str, now: float) -> float:
        """Record a failed attempt. Returns the delay now imposed."""
        count = self._failures.get(account_key, 0) + 1
        self._failures[account_key] = count
        delay = lockout_delay(count)
        if delay > 0:
            self._locked_until[account_key] = now + delay
        return delay

    def record_success(self, account_key: str) -> None:
        """Clear history after a successful login."""
        self._failures.pop(account_key, None)
        self._locked_until.pop(account_key, None)

    def failure_count(self, account_key: str) -> int:
        return self._failures.get(account_key, 0)
