"""Password hashing with Argon2id.

Argon2id, not bcrypt and not SHA-anything. bcrypt silently truncates at 72 bytes and has
no memory-hardness; a raw SHA is trivially GPU-parallelisable. Argon2id won the Password
Hashing Competition and is memory-hard, which is what makes offline cracking expensive.

`verify_password_constant_time` exists specifically to prevent user enumeration. The
naive shape of a login handler is:

    user = lookup(email)
    if user is None:
        return "invalid credentials"      # returns in microseconds
    if not verify(password, user.hash):
        return "invalid credentials"      # returns in ~50ms

Both branches say the same thing, but the timing does not, and the difference is large
enough to measure over a network. An attacker learns which addresses have accounts. The
fix is to verify against a dummy hash when the user does not exist, so both paths do the
same work.
"""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

# argon2-cffi's defaults track the library's own recommendations and are raised over
# time by its maintainers. Pinning our own numbers here would freeze them at whatever
# was current the day this was written, so the library's profile is used as-is and the
# resulting parameters are recorded in the hash string itself.
_hasher = PasswordHasher()

# Minimum length only. No composition rules: forcing a symbol and a digit measurably
# pushes people toward predictable patterns without adding real entropy.
MIN_PASSWORD_LENGTH = 12
# Argon2 has no bcrypt-style truncation, but an unbounded password is a cheap DoS -
# hashing a 10MB string is expensive by design.
MAX_PASSWORD_LENGTH = 1024

# A real Argon2id hash of a value nobody knows, used to burn equivalent CPU time when an
# account does not exist. Computed once at import.
_DUMMY_HASH = _hasher.hash("tayr-nonexistent-account-timing-equaliser")


class PasswordPolicyError(ValueError):
    """The supplied password does not meet policy."""


def validate_password_policy(password: str) -> None:
    """Raise if the password is unacceptable. Length only, by design."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"password must be at most {MAX_PASSWORD_LENGTH} characters")


def hash_password(password: str) -> str:
    """Hash a password with Argon2id. The salt and parameters live in the returned string."""
    validate_password_policy(password)
    return _hasher.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    """Check a password against a stored hash.

    Returns False on mismatch or on a malformed stored hash. It does not raise, because
    a caller that has to distinguish those cases tends to leak the distinction to the
    client.
    """
    try:
        return _hasher.verify(stored_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def verify_password_constant_time(password: str, stored_hash: str | None) -> bool:
    """Verify, doing equivalent work when the account does not exist.

    Pass `stored_hash=None` for an unknown user. The dummy verification takes the same
    order of time as a real one, so response timing does not reveal whether the address
    is registered.
    """
    if stored_hash is None:
        # Burn equivalent CPU against a real hash so the timing matches the found-user
        # path. The result is discarded; it is always False by construction.
        verify_password(password, _DUMMY_HASH)
        return False
    return verify_password(password, stored_hash)


def needs_rehash(stored_hash: str) -> bool:
    """True if the hash was made with weaker parameters than the current profile.

    Call after a successful login and re-hash if it returns True: that is how a
    deployment picks up the library's stronger defaults without a password reset.
    """
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except InvalidHashError:
        return True
