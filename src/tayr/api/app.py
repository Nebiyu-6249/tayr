"""FastAPI application factory.

Thin by design: routes call into the `tayr` library, and no business logic lives in a
handler. Middleware order matters and is commented where it does.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from tayr import __version__
from tayr.api import routes_auth, routes_videos
from tayr.api.deps import configure_database
from tayr.api.security_headers import add_security_headers
from tayr.api.settings import ApiSettings
from tayr.errors import TayrError
from tayr.security.ratelimit import RateLimitedError


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_database(app.state.settings.database_url)
    yield


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()

    app = FastAPI(
        title="Tayr API",
        version=__version__,
        description=(
            "Detection, tracking and motion classification of small aerial objects in "
            "recorded video."
        ),
        lifespan=_lifespan,
        # Interactive docs are off by default: they enumerate every endpoint and schema
        # for anyone who finds them. Turn on deliberately in development.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    # Per-application, not per-module: module-level limiters would be shared by every
    # app in the process. See routes_auth.AuthLimiters.
    app.state.auth_limiters = routes_auth.AuthLimiters()

    # Explicit origin allowlist, validated to reject "*" in ApiSettings. The Origin
    # header is never reflected: reflecting it with credentials enabled is equivalent
    # to having no CORS policy at all.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-CSRF-Token"],
        max_age=600,
    )
    app.middleware("http")(add_security_headers)

    @app.exception_handler(RateLimitedError)
    async def _rate_limited(request: Request, exc: RateLimitedError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "too many requests"},
            headers={"Retry-After": str(max(1, int(exc.retry_after_seconds)))},
        )

    @app.exception_handler(TayrError)
    async def _tayr_error(request: Request, exc: TayrError) -> JSONResponse:
        """Library errors become 400s with their own message.

        Anything NOT deriving from TayrError falls through to the default 500 handler,
        which returns no detail. That asymmetry is deliberate: TayrError messages are
        written to be shown to a user, while an arbitrary exception's message may carry
        a path, a query, or a stack frame.
        """
        del request
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(routes_auth.router)
    app.include_router(routes_videos.router)
    return app
