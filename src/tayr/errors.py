"""Explicit error types.

Tayr fails loudly. There is no `except: pass` anywhere in this codebase, no silent
fallback to a degraded path, and no function that returns an empty result to paper over
a failure. If something cannot be done, it raises. See CLAUDE.md "Failure policy".
"""


class TayrError(Exception):
    """Base class for every error Tayr raises deliberately."""


class ConfigError(TayrError):
    """A config file is missing, malformed, or fails schema validation."""


class GeometryError(TayrError):
    """A bounding box or coordinate transform received invalid input.

    Raised rather than silently clamping. A negative-width box is a bug in a converter,
    and clamping it to zero would hide that bug behind a plausible-looking number.
    """


class ManifestError(TayrError):
    """A run manifest could not be built - usually because git metadata is unavailable."""


class DependencyUnavailableError(TayrError):
    """An optional dependency needed for the requested operation is not installed.

    Raised instead of falling back. A missing torch must stop a training run, not
    silently produce results from some other code path.
    """
