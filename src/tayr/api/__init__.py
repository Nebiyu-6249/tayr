"""FastAPI application. A thin wrapper over the `tayr` library."""

from tayr.api.app import create_app
from tayr.api.settings import ApiSettings

__all__ = ["ApiSettings", "create_app"]
