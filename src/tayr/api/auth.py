"""Session authentication, CSRF, and per-record authorisation.

Cookie design:

  `__Host-tayr_session`  The `__Host-` prefix is a browser-enforced guarantee: the
                         cookie must be Secure, must have Path=/, and must have **no
                         Domain attribute**. That last part is what stops a subdomain
                         (including one an attacker gets control of) from setting a
                         session cookie the main site would honour.
  HttpOnly               JavaScript cannot read it, so an XSS that gets past CSP still
                         cannot exfiltrate the session.
  SameSite=Lax           The browser will not send it on cross-site POSTs, which is the
                         first line against CSRF. Lax rather than Strict so that
                         following a link from elsewhere does not appear logged out.

CSRF tokens back that up with the double-submit pattern, because SameSite is a browser
behaviour and not every client enforces it identically.

Authorisation is checked per record, on reads and writes alike. `require_owned` is the
single place that check lives, so a new endpoint gets it by using the dependency rather
than by remembering to write a filter.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Cookie, Depends, Header, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tayr.db.models import (
    AgentDecisionRecord,
    Job,
    Session,
    TrackRecord,
    User,
    Video,
)
from tayr.security.tokens import (
    constant_time_compare,
    generate_csrf_token,
    generate_session_token,
    hash_token,
)

SESSION_COOKIE = "__Host-tayr_session"
CSRF_COOKIE = "__Host-tayr_csrf"
CSRF_HEADER = "X-CSRF-Token"

SESSION_LIFETIME = timedelta(days=7)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# A single opaque message for every authentication failure. Distinguishing "no session"
# from "expired session" from "revoked session" tells an attacker which of those they
# achieved.
_UNAUTHENTICATED = "authentication required"


async def create_session(db: AsyncSession, user: User, response: Response) -> str:
    """Issue a session, set both cookies, and return the CSRF token."""
    token = generate_session_token()
    csrf = generate_csrf_token()

    db.add(
        Session(
            user_id=user.id,
            token_hash=hash_token(token),
            csrf_token_hash=hash_token(csrf),
            expires_at=datetime.now(UTC) + SESSION_LIFETIME,
            epoch=user.session_epoch,
        )
    )
    await db.flush()

    max_age = int(SESSION_LIFETIME.total_seconds())
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=max_age,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    # Readable by JavaScript on purpose: the client must echo it in a header for the
    # double-submit check. It is not a credential on its own.
    response.set_cookie(
        CSRF_COOKIE,
        csrf,
        max_age=max_age,
        httponly=False,
        secure=True,
        samesite="lax",
        path="/",
    )
    return csrf


def clear_session_cookies(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")


async def get_current_user(
    request: Request,
    db: AsyncSession,
    session_token: str | None,
) -> User:
    """Resolve the session cookie to a user, or raise 401.

    Four things invalidate a session, and all four return the same message:
    no cookie, no matching row, expiry, and an epoch mismatch (password changed).
    """
    del request
    if not session_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _UNAUTHENTICATED)

    result = await db.execute(
        select(Session).where(Session.token_hash == hash_token(session_token))
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _UNAUTHENTICATED)

    expires_at = session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at < datetime.now(UTC):
        await db.delete(session)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _UNAUTHENTICATED)

    user = await db.get(User, session.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _UNAUTHENTICATED)

    # Password change bumps the epoch, invalidating every session issued before it.
    if session.epoch != user.session_epoch:
        await db.delete(session)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _UNAUTHENTICATED)

    return user


async def verify_csrf(
    request: Request,
    db: AsyncSession,
    session_token: str | None,
    csrf_header: str | None,
) -> None:
    """Double-submit CSRF check on state-changing requests.

    The header value is compared against the hash stored server-side, not merely against
    the cookie. A pure cookie-vs-header comparison is defeated by any attacker who can
    set a cookie on the domain.
    """
    if request.method in _SAFE_METHODS:
        return
    if not csrf_header or not session_token:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "CSRF token missing")

    result = await db.execute(
        select(Session).where(Session.token_hash == hash_token(session_token))
    )
    session = result.scalar_one_or_none()
    if session is None or not constant_time_compare(
        hash_token(csrf_header), session.csrf_token_hash
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "CSRF token invalid")


async def require_owned[OwnedT: (Video, Job, TrackRecord, AgentDecisionRecord)](
    db: AsyncSession, model: type[OwnedT], record_id: str, user: User
) -> OwnedT:
    """Fetch a record the user owns, or raise 404.

    **404, not 403.** Returning 403 for an existing record owned by someone else
    confirms that the id exists, which turns id enumeration into a census of other
    users' data. A caller cannot distinguish "does not exist" from "not yours".

    This is the only place an ownership check is written. New endpoints get it by using
    this function rather than by remembering to add a filter.
    """
    record = await db.get(model, record_id)
    if record is None or record.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    return record


SessionCookie = Annotated[str | None, Cookie(alias=SESSION_COOKIE)]
CsrfHeader = Annotated[str | None, Header(alias=CSRF_HEADER)]
__all__ = [
    "CSRF_COOKIE",
    "CSRF_HEADER",
    "SESSION_COOKIE",
    "SESSION_LIFETIME",
    "CsrfHeader",
    "Depends",
    "SessionCookie",
    "clear_session_cookies",
    "create_session",
    "get_current_user",
    "require_owned",
    "verify_csrf",
]
