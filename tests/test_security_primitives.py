"""Security primitive tests.

Each control is tested on its own rather than only through an endpoint, so a regression
points at the specific defence that broke.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from tayr.security.passwords import (
    MIN_PASSWORD_LENGTH,
    PasswordPolicyError,
    hash_password,
    needs_rehash,
    verify_password,
    verify_password_constant_time,
)
from tayr.security.ratelimit import (
    LoginAttemptTracker,
    RateLimitedError,
    RateLimiter,
    TokenBucket,
    lockout_delay,
)
from tayr.security.tokens import (
    constant_time_compare,
    generate_csrf_token,
    generate_session_token,
    hash_token,
)
from tayr.security.uploads import (
    MediaProperties,
    UploadRejectedError,
    VideoLimits,
    detect_container,
    enforce_media_limits,
    generate_storage_name,
    is_safe_storage_path,
    validate_upload,
)

# S105 suppressed: this is test fixture data, not a credential. The rule stays on
# everywhere else, where a hardcoded password would be a real finding.
GOOD_PASSWORD = "correct-horse-battery-staple"  # noqa: S105


class TestPasswords:
    def test_hash_verify_round_trip(self) -> None:
        assert verify_password(GOOD_PASSWORD, hash_password(GOOD_PASSWORD))

    def test_wrong_password_rejected(self) -> None:
        assert not verify_password("wrong-password-entirely", hash_password(GOOD_PASSWORD))

    def test_uses_argon2id(self) -> None:
        """Not bcrypt (72-byte truncation, no memory hardness), not SHA (GPU-cheap)."""
        assert hash_password(GOOD_PASSWORD).startswith("$argon2id$")

    def test_salted_so_identical_passwords_differ(self) -> None:
        """Unsalted hashes let one rainbow table crack every account at once."""
        assert hash_password(GOOD_PASSWORD) != hash_password(GOOD_PASSWORD)

    def test_long_password_not_truncated(self) -> None:
        """bcrypt would treat these two as the same password; Argon2 does not."""
        a = "x" * 100 + "aaaa"
        b = "x" * 100 + "bbbb"
        assert not verify_password(b, hash_password(a))

    def test_malformed_stored_hash_returns_false_rather_than_raising(self) -> None:
        assert not verify_password(GOOD_PASSWORD, "not-a-hash")

    @pytest.mark.parametrize("bad", ["short", "x" * (MIN_PASSWORD_LENGTH - 1)])
    def test_policy_rejects_short_passwords(self, bad: str) -> None:
        with pytest.raises(PasswordPolicyError, match="at least"):
            hash_password(bad)

    def test_policy_rejects_absurdly_long_passwords(self) -> None:
        """Unbounded input is a cheap denial of service against a memory-hard hash."""
        with pytest.raises(PasswordPolicyError, match="at most"):
            hash_password("x" * 5000)

    def test_needs_rehash_is_false_for_a_current_hash(self) -> None:
        assert needs_rehash(hash_password(GOOD_PASSWORD)) is False

    def test_needs_rehash_is_true_for_garbage(self) -> None:
        assert needs_rehash("not-a-hash") is True

    def test_unknown_user_path_does_comparable_work(self) -> None:
        """No user enumeration: a missing account must not return measurably faster than
        a wrong password, or an attacker learns which addresses are registered."""
        stored = hash_password(GOOD_PASSWORD)

        def timed(fn: Callable[[], object], n: int = 5) -> float:
            start = time.perf_counter()
            for _ in range(n):
                fn()
            return (time.perf_counter() - start) / n

        wrong_pw = timed(lambda: verify_password_constant_time("nope-not-this-one", stored))
        no_user = timed(lambda: verify_password_constant_time("nope-not-this-one", None))
        assert verify_password_constant_time("x", None) is False
        # Generous bound: this asserts the same order of magnitude, not a tight timing
        # guarantee, because CI machines are noisy.
        assert 0.2 < (no_user / wrong_pw) < 5.0, f"ratio {no_user / wrong_pw:.2f}"


class TestTokens:
    def test_tokens_are_unique(self) -> None:
        assert len({generate_session_token() for _ in range(500)}) == 500

    def test_tokens_have_real_entropy(self) -> None:
        """32 bytes url-safe base64 is at least 43 characters."""
        assert len(generate_session_token()) >= 43
        assert len(generate_csrf_token()) >= 43

    def test_hash_token_is_stable_and_one_way(self) -> None:
        token = generate_session_token()
        assert hash_token(token) == hash_token(token)
        assert token not in hash_token(token)
        assert len(hash_token(token)) == 64

    def test_different_tokens_hash_differently(self) -> None:
        assert hash_token(generate_session_token()) != hash_token(generate_session_token())

    def test_constant_time_compare(self) -> None:
        t = generate_session_token()
        assert constant_time_compare(t, t)
        assert not constant_time_compare(t, generate_session_token())
        assert not constant_time_compare(t, t[:-1] + ("A" if t[-1] != "A" else "B"))

    def test_secrets_not_random(self) -> None:
        """`random` is a Mersenne Twister: a few hundred outputs reveal its state and
        all future ones. Seeding it must not make tokens predictable."""
        import random

        random.seed(42)
        first = generate_session_token()
        random.seed(42)
        assert generate_session_token() != first


class TestContainerDetection:
    def test_mp4_by_magic_bytes(self) -> None:
        assert detect_container(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 16) == "mp4"

    def test_matroska(self) -> None:
        assert detect_container(b"\x1a\x45\xdf\xa3" + b"\x00" * 16) == "matroska"

    def test_avi_requires_both_markers(self) -> None:
        assert detect_container(b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 8) == "avi"
        # RIFF alone is also WAV, WebP and others.
        assert detect_container(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 8) is None

    def test_extension_is_never_trusted(self) -> None:
        """A file named evil.mp4 whose contents are a script must be rejected."""
        assert detect_container(b"#!/bin/sh\nrm -rf /\n" + b"\x00" * 8) is None

    def test_truncated_header(self) -> None:
        assert detect_container(b"\x00\x00") is None


class TestUploadValidation:
    MP4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 16

    def test_accepts_a_real_mp4(self) -> None:
        assert validate_upload(self.MP4, 1_000_000) == "mp4"

    def test_rejects_oversize(self) -> None:
        with pytest.raises(UploadRejectedError, match="limit"):
            validate_upload(self.MP4, 10**12)

    def test_rejects_empty(self) -> None:
        with pytest.raises(UploadRejectedError, match="empty"):
            validate_upload(self.MP4, 0)

    def test_rejects_unrecognised_content(self) -> None:
        with pytest.raises(UploadRejectedError, match="unrecognised video container"):
            validate_upload(b"GIF89a" + b"\x00" * 20, 5000)

    def test_storage_name_is_generated_not_user_supplied(self) -> None:
        """The user's filename is the classic path-traversal vector and can leak PII."""
        a, b = generate_storage_name("mp4"), generate_storage_name("mp4")
        assert a != b
        assert a.endswith(".mp4")
        for bad in ("..", "/", "\\", "\x00"):
            assert bad not in a

    def test_storage_name_rejects_unknown_container(self) -> None:
        with pytest.raises(UploadRejectedError, match="unsupported container"):
            generate_storage_name("exe")

    def test_safe_storage_path(self, tmp_path: Path) -> None:
        root = tmp_path / "uploads"
        root.mkdir()
        assert is_safe_storage_path(root / "abc.mp4", root=root)
        assert not is_safe_storage_path(root / ".." / ".." / "etc" / "passwd", root=root)


class TestMediaLimits:
    def test_accepts_normal_video(self) -> None:
        enforce_media_limits(MediaProperties(1920, 1080, 30.0, 60.0))

    def test_rejects_excessive_resolution(self) -> None:
        with pytest.raises(UploadRejectedError, match="resolution"):
            enforce_media_limits(MediaProperties(15360, 8640, 30.0, 10.0))

    def test_rejects_excessive_duration(self) -> None:
        with pytest.raises(UploadRejectedError, match="duration"):
            enforce_media_limits(MediaProperties(640, 480, 30.0, 100_000.0))

    def test_rejects_excessive_fps(self) -> None:
        with pytest.raises(UploadRejectedError, match="frame rate"):
            enforce_media_limits(MediaProperties(640, 480, 10_000.0, 5.0))

    def test_rejects_a_decompression_bomb_whose_parts_are_all_legal(self) -> None:
        """The attack this control exists for: every individual property is inside its
        cap, but the product is enormous. A small file, unbounded RAM."""
        bomb = MediaProperties(width=4096, height=4096, fps=120.0, duration_seconds=299.0)
        assert bomb.width <= VideoLimits().max_width
        assert bomb.height <= VideoLimits().max_height
        assert bomb.fps <= VideoLimits().max_fps
        assert bomb.duration_seconds <= VideoLimits().max_duration_seconds
        with pytest.raises(UploadRejectedError, match="decompression bomb"):
            enforce_media_limits(bomb)

    @pytest.mark.parametrize(
        "props",
        [
            MediaProperties(0, 1080, 30.0, 10.0),
            MediaProperties(1920, -1, 30.0, 10.0),
            MediaProperties(1920, 1080, 0.0, 10.0),
            MediaProperties(1920, 1080, 30.0, 0.0),
        ],
    )
    def test_rejects_nonsense_properties(self, props: MediaProperties) -> None:
        with pytest.raises(UploadRejectedError, match="invalid"):
            enforce_media_limits(props)


class TestRateLimiting:
    def test_allows_up_to_capacity_then_blocks(self) -> None:
        bucket = TokenBucket(capacity=3, refill_per_second=1.0)
        for _ in range(3):
            bucket.consume(now=0.0)
        with pytest.raises(RateLimitedError):
            bucket.consume(now=0.0)

    def test_refills_over_time(self) -> None:
        bucket = TokenBucket(capacity=3, refill_per_second=1.0)
        for _ in range(3):
            bucket.consume(now=0.0)
        bucket.consume(now=1.0)  # one token back

    def test_retry_after_is_useful(self) -> None:
        bucket = TokenBucket(capacity=1, refill_per_second=0.5)
        bucket.consume(now=0.0)
        with pytest.raises(RateLimitedError) as exc:
            bucket.consume(now=0.0)
        assert exc.value.retry_after_seconds == pytest.approx(2.0)

    def test_keys_are_independent(self) -> None:
        clock = iter([0.0] * 20)
        limiter = RateLimiter(capacity=2, refill_per_second=1.0, clock=lambda: next(clock))
        limiter.check("user-a")
        limiter.check("user-a")
        limiter.check("user-b")  # different key, own bucket
        with pytest.raises(RateLimitedError):
            limiter.check("user-a")

    def test_no_burst_across_a_window_boundary(self) -> None:
        """A fixed-window limiter allows 2x the limit straddling a boundary. A token
        bucket does not, which is why it was chosen."""
        bucket = TokenBucket(capacity=5, refill_per_second=5.0 / 60.0)
        for _ in range(5):
            bucket.consume(now=59.0)
        with pytest.raises(RateLimitedError):
            bucket.consume(now=61.0)


class TestProgressiveLockout:
    def test_first_few_attempts_are_free(self) -> None:
        """A person who mistypes twice should not be punished."""
        assert lockout_delay(1) == 0.0
        assert lockout_delay(3) == 0.0

    def test_delay_grows_then_caps(self) -> None:
        delays = [lockout_delay(n) for n in range(4, 30)]
        assert delays[0] > 0
        assert all(b >= a for a, b in itertools.pairwise(delays))
        assert max(delays) == 900.0

    def test_tracker_locks_and_reports_retry_after(self) -> None:
        t = LoginAttemptTracker()
        for _ in range(6):
            t.record_failure("user@example.com", now=0.0)
        with pytest.raises(RateLimitedError) as exc:
            t.check("user@example.com", now=0.0)
        assert exc.value.retry_after_seconds > 0

    def test_success_clears_history(self) -> None:
        t = LoginAttemptTracker()
        for _ in range(6):
            t.record_failure("user@example.com", now=0.0)
        t.record_success("user@example.com")
        assert t.failure_count("user@example.com") == 0
        t.check("user@example.com", now=0.0)

    def test_lock_expires(self) -> None:
        t = LoginAttemptTracker()
        delay = 0.0
        for _ in range(5):
            delay = t.record_failure("u", now=0.0)
        t.check("u", now=delay + 0.001)

    def test_lockout_is_keyed_on_account_not_address(self) -> None:
        """Keying on address lets an attacker with many addresses guess freely, and
        locking an address shuts out everyone behind one NAT."""
        t = LoginAttemptTracker()
        for _ in range(6):
            t.record_failure("victim@example.com", now=0.0)
        t.check("someone-else@example.com", now=0.0)
