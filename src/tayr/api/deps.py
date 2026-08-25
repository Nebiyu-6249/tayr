"""FastAPI dependencies: database session, current user, CSRF."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tayr.api import auth
from tayr.api.settings import ApiSettings
from tayr.db.models import User

_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def configure_database(url: str) -> async_sessionmaker[AsyncSession]:
    """Create the engine and session factory. Called once at startup."""
    global _engine, _sessionmaker
    _engine = create_async_engine(url, future=True)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _sessionmaker


async def get_db() -> AsyncIterator[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("database not configured; call configure_database() at startup")
    async with _sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_settings(request: Request) -> ApiSettings:
    settings: ApiSettings = request.app.state.settings
    return settings


Settings = Annotated[ApiSettings, Depends(get_settings)]


async def current_user(
    request: Request, db: DbSession, session_token: auth.SessionCookie = None
) -> User:
    return await auth.get_current_user(request, db, session_token)


CurrentUser = Annotated[User, Depends(current_user)]


async def csrf_protected(
    request: Request,
    db: DbSession,
    session_token: auth.SessionCookie = None,
    csrf_header: auth.CsrfHeader = None,
) -> None:
    await auth.verify_csrf(request, db, session_token, csrf_header)


CsrfProtected = Annotated[None, Depends(csrf_protected)]
