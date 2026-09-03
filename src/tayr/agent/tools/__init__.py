"""Agent tools: the allowlist, the context, and the registered tool set."""

from tayr.agent.tools.context import HistoricalTrack, ToolContext, TrackSnapshot
from tayr.agent.tools.readonly import READ_ONLY_SPECS
from tayr.agent.tools.registry import (
    ToolArgumentError,
    ToolEffect,
    ToolRegistry,
    ToolSpec,
    UnknownToolError,
)


def build_registry(*, include_acting: bool = False) -> ToolRegistry:
    """Build the tool allowlist.

    Acting tools are opt-in. A read-only agent run cannot post to Slack or open an
    incident because those tools are not in its registry at all - not because it
    chose not to call them.
    """
    registry = ToolRegistry()
    for spec in READ_ONLY_SPECS:
        registry.register(spec)
    if include_acting:
        from tayr.agent.tools.acting import ACTING_SPECS

        for spec in ACTING_SPECS:
            registry.register(spec)
    return registry


__all__ = [
    "READ_ONLY_SPECS",
    "HistoricalTrack",
    "ToolArgumentError",
    "ToolContext",
    "ToolEffect",
    "ToolRegistry",
    "ToolSpec",
    "TrackSnapshot",
    "UnknownToolError",
    "build_registry",
]
