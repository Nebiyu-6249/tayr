"""Authentication endpoints.

The recurring theme here is that **registration and login must be indistinguishable to
an attacker probing for valid addresses**. Both return the same shape, the same status
code, and comparable timing regardless of whether the account exists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import delete, select

from tayr.api import auth
from tayr.api.deps import CsrfProtected, CurrentUser, DbSession
from tayr.api.schemas import (
    ChangePasswordRequest,
    ErrorOut,
    LoginRequest,
    RegisterRequest,
    UserOut,
)
from tayr.db.models import Session, User
from tayr.security.passwords import (
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
)

router = APIRouter(prefix="/auth", tags=["auth"])


@dataclass
class AuthLimiters:
    """Rate-limit state, held on `app.state` rather than at module scope.

    Module-level globals would be shared by every application in the process, which
    makes the limits untestable and couples unrelated app instances. They are also
    per-process: a deployment running several API workers gets N times the intended
    limit, because each worker holds its own buckets. That is recorded as an accepted
    risk in docs/THREAT_MODEL.md, and the fix is a Redis-backed limiter.
    """

    login: RateLimiter = field(
        default_factory=lambda: RateLimiter(capacity=10, refill_per_second=10 / 300)
    )
    register: RateLimiter = field(
        default_factory=lambda: RateLimiter(capacity=5, refill_per_second=5 / 3600)
    )
    attempts: LoginAttemptTracker = field(default_factory=LoginAttemptTracker)


def _limiters(request: Request) -> AuthLimiters:
    limiters: AuthLimiters = request.app.state.auth_limiters
    return limiters


# One message for every credential failure. "No such user" and "wrong password" must be
# indistinguishable, or the endpoint becomes an account-existence oracle.
_INVALID_CREDENTIALS = "invalid email or password"

# Deliberately non-committal: it does not say whether an account was created.
_REGISTRATION_ACCEPTED = (
    "if that address was available, an account has been created; sign in to continue"
)


def _client_key(request: Request) -> str:
    client = request.client
    return client.host if client else "unknown"


@router.post("/register", response_model=ErrorOut, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, request: Request, db: DbSession) -> ErrorOut:
    """Create an account.

    NO ENUMERATION. An address that is already registered gets byte-identical output to
    a new one: the same 201, the same body, and comparable latency (the password is
    hashed before the lookup either way, so the expensive operation happens on both
    paths). A 409 here would turn this endpoint into an account-existence oracle, and
    rate limiting alone would not fix that - an attacker only needs one request per
    address they care about.

    Two consequences, both deliberate:

      - **No session is issued.** Returning one would have to either succeed or fail
        differently for a duplicate, reintroducing the oracle. The client signs in as a
        separate step.
      - **No user object is returned.** Returning the existing user's record for a
        duplicate address would be considerably worse than merely confirming it exists.

    The usual way to keep enumeration closed *and* give good feedback is to send a mail
    ("someone tried to register with your address"). Tayr has no mail delivery, so this
    endpoint takes the conservative half of that trade and the UX cost is accepted. It
    is recorded in docs/THREAT_MODEL.md.
    """
    try:
        _limiters(request).register.check(_client_key(request))
    except RateLimitedError as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "too many registration attempts",
            headers={"Retry-After": str(int(exc.retry_after_seconds))},
        ) from exc

    email = payload.email.lower().strip()
    try:
        password_hash = hash_password(payload.password)
    except PasswordPolicyError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    existing = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if existing is None:
        db.add(User(email=email, password_hash=password_hash))
        await db.flush()

    # Identical on both paths. The caller cannot tell which one ran.
    return ErrorOut(detail=_REGISTRATION_ACCEPTED)


@router.post("/login", response_model=UserOut)
async def login(payload: LoginRequest, request: Request, response: Response, db: DbSession) -> User:
    """Start a session. Identical response for unknown user and wrong password."""
    email = payload.email.lower().strip()
    now = time.monotonic()

    try:
        limiters = _limiters(request)
        limiters.login.check(_client_key(request))
        limiters.attempts.check(email, now)
    except RateLimitedError as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "too many attempts",
            headers={"Retry-After": str(max(1, int(exc.retry_after_seconds)))},
        ) from exc

    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()

    # Verifies against a dummy hash when the user is absent, so both paths cost the same.
    ok = verify_password_constant_time(
        payload.password, user.password_hash if user is not None else None
    )
    if not ok or user is None or not user.is_active:
        _limiters(request).attempts.record_failure(email, now)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _INVALID_CREDENTIALS)

    _limiters(request).attempts.record_success(email)

    # Pick up stronger Argon2 parameters without forcing a reset.
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)

    await auth.create_session(db, user, response)
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response, db: DbSession, _: CsrfProtected, session_token: auth.SessionCookie = None
) -> None:
    """End this session. Deletes the row, not just the cookie."""
    if session_token:
        from tayr.security.tokens import hash_token

        result = await db.execute(
            select(Session).where(Session.token_hash == hash_token(session_token))
        )
        row = result.scalar_one_or_none()
        if row is not None:
            await db.delete(row)
    auth.clear_session_cookies(response)


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    payload: ChangePasswordRequest, db: DbSession, user: CurrentUser, _: CsrfProtected
) -> None:
    """Change the password and invalidate every existing session.

    Bumping `session_epoch` logs out every other device. If an attacker had a stolen
    session, changing the password is the user's remedy - and it only works if it
    actually revokes that session.
    """
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _INVALID_CREDENTIALS)
    try:
        user.password_hash = hash_password(payload.new_password)
    except PasswordPolicyError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    user.session_epoch += 1
    await db.execute(delete(Session).where(Session.user_id == user.id))


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> User:
    return user
