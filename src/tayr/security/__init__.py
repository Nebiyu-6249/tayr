"""Security primitives.

Deliberately dependency-light and free of any web framework, so each control can be
unit-tested on its own rather than only through an endpoint. Everything here is used by
`tayr.api`; nothing here imports it.
"""

from tayr.security.passwords import (
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

__all__ = [
    "LoginAttemptTracker",
    "MediaProperties",
    "RateLimitedError",
    "RateLimiter",
    "TokenBucket",
    "UploadRejectedError",
    "VideoLimits",
    "constant_time_compare",
    "detect_container",
    "enforce_media_limits",
    "generate_csrf_token",
    "generate_session_token",
    "generate_storage_name",
    "hash_password",
    "hash_token",
    "is_safe_storage_path",
    "lockout_delay",
    "needs_rehash",
    "validate_upload",
    "verify_password",
    "verify_password_constant_time",
]
