"""LLM layer: natural-language incident summaries from structured detection output.

Narrow by design. The model turns an already-decided track record into readable prose.
It makes no classification decision, it never sees pixels, and the system produces full
structured results without it.
"""

from tayr.llm.provider import (
    IncidentSummary,
    LLMProvider,
    LLMUnavailableError,
    NullProvider,
    TrackSummaryInput,
    UsageRecord,
)
from tayr.llm.summariser import Summariser

__all__ = [
    "IncidentSummary",
    "LLMProvider",
    "LLMUnavailableError",
    "NullProvider",
    "Summariser",
    "TrackSummaryInput",
    "UsageRecord",
]
