"""Provider interface and structured types.

The provider sits behind a Protocol so OpenAI can be swapped without touching callers.
Model ids are `[VERIFIED]` against the OpenAI SDK's generated
`openai/types/shared/chat_model.py`; **pricing is `[UNKNOWN]`** because the pricing page
was unreachable during Phase 0, so the cost cap is expressed in tokens as well as
dollars and the dollar figure must be confirmed before it is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from tayr.errors import TayrError


class LLMUnavailableError(TayrError):
    """The provider could not be reached, or refused. Never fatal to a job."""


class IncidentSummary(BaseModel):
    """Schema the model must return. Validated on the way back, not trusted.

    A model that returns prose where a float was expected, or invents a class outside
    the enum, fails validation and the summary is dropped. It never becomes a value the
    rest of the system acts on.
    """

    model_config = ConfigDict(extra="forbid")

    headline: str = Field(max_length=200)
    narrative: str = Field(max_length=2000)
    notable_tracks: list[int] = Field(default_factory=list, max_length=50)


@dataclass(frozen=True, slots=True)
class TrackSummaryInput:
    """The structured record handed to the model. Numbers only, no free text.

    Deliberately not a dict of arbitrary keys: everything here is produced by Tayr's own
    pipeline, so nothing an uploader wrote can reach the prompt through this path.
    """

    track_number: int
    label: str
    confidence: float
    first_frame: int
    last_frame: int
    median_pixels_on_target: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Token counts for cost tracking. Logged on every call."""

    model: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    """What the summariser requires of a provider."""

    @property
    def model(self) -> str: ...

    def complete(self, *, system: str, user: str, max_tokens: int) -> tuple[str, UsageRecord]:
        """Return the model's raw text and its token usage.

        Raises `LLMUnavailableError` on any provider failure. Callers degrade; they do
        not propagate.
        """
        ...


class NullProvider:
    """A provider that is always unavailable.

    The default. Tayr produces complete structured results with no LLM configured, and
    this makes that the path taken by default rather than an untested fallback.
    """

    @property
    def model(self) -> str:
        return "null"

    def complete(self, *, system: str, user: str, max_tokens: int) -> tuple[str, UsageRecord]:
        del system, user, max_tokens
        raise LLMUnavailableError("no LLM provider is configured")
