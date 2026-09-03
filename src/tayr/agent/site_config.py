"""Site configuration: airspace posture and the authorized-flight registry.

Seeded from version-controlled YAML, not a live API. That is deliberate for three
reasons: a live airspace feed is an external dependency the demo cannot rely on, an
outbound call from the agent would be a network egress the worker sandbox forbids, and
a config file is auditable in a way an API response is not.

**The authorized-flight registry is the single most valuable tool the agent has.** Most
detected drones are somebody's permitted flight, so this is where most dismissals come
from - and dismissals are the product. It is also the only tool that can produce a
confident DISMISS in a deployment with no trained classifier, which is the current
state.

HONEST LIMITATION: matching is on **site and time window only**. A real deployment would
also match position and altitude against the filed flight envelope, which needs
georeferenced tracks Tayr does not produce. As it stands, any drone airborne during an
authorized window at that site matches - which is permissive, and would be wrong in a
real deployment. `AuthorizationMatch.match_basis` records this so the limitation appears
in the audit record rather than living only in this docstring.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tayr.errors import ConfigError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuthorizedFlight(_Strict):
    """One filed, permitted flight."""

    flight_id: str = Field(max_length=64)
    operator: str = Field(max_length=128)
    """Display text only. Escaped on output; never used for authorisation."""

    starts_at: datetime
    ends_at: datetime
    notes: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _window_ordered(self) -> AuthorizedFlight:
        if self.ends_at <= self.starts_at:
            raise ValueError(
                f"flight {self.flight_id}: ends_at must be after starts_at "
                f"({self.ends_at} <= {self.starts_at})"
            )
        return self

    def covers(self, moment: datetime) -> bool:
        return self.starts_at <= moment <= self.ends_at


class TemporaryRestriction(_Strict):
    """An active restriction over this site for a period."""

    restriction_id: str = Field(max_length=64)
    reason: str = Field(max_length=200)
    starts_at: datetime
    ends_at: datetime

    def covers(self, moment: datetime) -> bool:
        return self.starts_at <= moment <= self.ends_at


ZoneClass = Literal["unrestricted", "controlled", "restricted"]


class SiteConfig(_Strict):
    """One protected site.

    `zone_class` describes the airspace posture, which changes how much an unexplained
    detection matters: an unidentified object over an unrestricted field is routine, the
    same object over a restricted site is not. It does not describe any response, and
    nothing downstream may treat it as one.
    """

    site_id: str = Field(max_length=64)
    name: str = Field(max_length=128)
    zone_class: ZoneClass = "unrestricted"
    authorized_flights: tuple[AuthorizedFlight, ...] = ()
    restrictions: tuple[TemporaryRestriction, ...] = ()


class SiteRegistry(_Strict):
    """All configured sites."""

    sites: tuple[SiteConfig, ...] = ()

    def get(self, site_id: str) -> SiteConfig | None:
        return next((s for s in self.sites if s.site_id == site_id), None)


def load_site_registry(path: str | Path) -> SiteRegistry:
    """Load and validate the site registry. Raises ConfigError with the path attached."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"site registry not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{p} must contain a YAML mapping at the top level")
    try:
        return SiteRegistry.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"{p} failed validation:\n{exc}") from exc


class AuthorizationMatch(_Strict):
    """The result of an authorization check."""

    authorized: bool
    flight_id: str | None = None
    operator: str | None = None
    match_basis: str = "site_and_time_window"
    """How the match was made. Recorded in the audit trail because the basis is coarse:
    site and time only, not position or altitude. An auditor must be able to see that."""

    checked_at: datetime | None = None
    site_known: bool = True


class ZoneStatus(_Strict):
    """The protection posture at a place and time."""

    site_id: str
    site_known: bool
    zone_class: ZoneClass = "unrestricted"
    restriction_active: bool = False
    restriction_reason: str | None = None


PreSeconds = Annotated[float, Field(ge=0.0, le=30.0)]
PostSeconds = Annotated[float, Field(ge=0.0, le=30.0)]
