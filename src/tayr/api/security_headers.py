"""Security response headers and CORS.

Each header here closes a specific attack, noted inline. They are applied by middleware
rather than per-route, because a header that has to be remembered on every new endpoint
is one that will eventually be forgotten.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

# One year, the minimum for preload-list eligibility.
_HSTS_MAX_AGE = 31_536_000

SECURITY_HEADERS: dict[str, str] = {
    # No inline script or style. This is what makes an injected <script> inert, and it
    # is why the frontend must not use inline handlers or styled attributes.
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    ),
    # Clickjacking. frame-ancestors above covers modern browsers; this covers the rest.
    "X-Frame-Options": "DENY",
    # Stops a browser from MIME-sniffing an upload into something executable.
    "X-Content-Type-Options": "nosniff",
    # Keeps paths and ids out of the Referer header on outbound links.
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # This application needs none of these.
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    # Blocks cross-origin resource loading of API responses.
    "Cross-Origin-Resource-Policy": "same-origin",
}


class SecurityHeadersMiddleware:
    """Applies the header set to every response.

    HSTS is only sent over HTTPS: sending it on a plaintext development request would
    pin localhost to HTTPS in the developer's browser, which is a memorable afternoon.
    """

    def __init__(self, app: ASGIApp, *, enable_hsts: bool = True) -> None:
        self.app = app
        self.enable_hsts = enable_hsts

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message) -> None:  # type: ignore[no-untyped-def]
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                for key, value in SECURITY_HEADERS.items():
                    headers.append((key.lower().encode(), value.encode()))
                if self.enable_hsts and scope.get("scheme") == "https":
                    headers.append(
                        (
                            b"strict-transport-security",
                            f"max-age={_HSTS_MAX_AGE}; includeSubDomains; preload".encode(),
                        )
                    )
            await send(message)

        await self.app(scope, receive, send_with_headers)


async def add_security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Function-style equivalent, for use with `app.middleware('http')`."""
    response = await call_next(request)
    for key, value in SECURITY_HEADERS.items():
        response.headers[key] = value
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = (
            f"max-age={_HSTS_MAX_AGE}; includeSubDomains; preload"
        )
    return response
