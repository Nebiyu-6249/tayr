"""Tool registry: the allowlist, and the only path by which a model can reach code.

**This is the privilege boundary.** In the summariser, a successful prompt injection got
an attacker a differently-worded paragraph. An agent with tools is a different risk
class: injected text now sits in the context of a loop that can call
`escalate_to_human` and `open_incident`. Four properties hold that line, and all four
are enforced here rather than by the model behaving well:

1. **An unregistered tool name is a hard error**, logged as a security event, never a
   retry. A model that names `run_shell` gets an exception, not another turn to try
   again with better spelling.
2. **Arguments are Pydantic-validated** against the tool's own schema before the
   function is called. The OpenAI SDK itself warns that a model "may hallucinate
   parameters not defined by your function schema" and says to validate in your code
   `[VERIFIED: openai 3.3.1, chat_completion_message_function_tool_call.py]`.
3. **Acting tools are separated from read-only ones** and counted separately, so a cap
   on actions is a cap on actions regardless of how much reading the model does.
4. **No tool deletes anything, modifies a model, changes configuration, executes a
   command, or reaches the public internet.** That is a property of the set, asserted by
   `test_agent_tools.py::test_no_tool_can_act_outside_its_declared_effect`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError

from tayr.agent.records import ToolInvocation
from tayr.errors import TayrError
from tayr.security.audit import SecurityEvent, log_security_event


class ToolEffect(StrEnum):
    """Whether a tool can change anything. Declared per tool, never inferred."""

    READ_ONLY = "read_only"
    ACTING = "acting"
    """Changes state outside the agent: posts a notification, opens an incident.
    There are exactly two, and both are idempotent per track."""


class UnknownToolError(TayrError):
    """The model named a tool that is not registered. Never retried."""


class ToolArgumentError(TayrError):
    """Arguments failed validation. Returned to the model as a tool error, not raised
    to the caller: a malformed call is something the model can correct, unlike naming a
    tool that does not exist."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One registered tool."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any, BaseModel], dict[str, Any]]
    effect: ToolEffect

    def openai_schema(self) -> dict[str, Any]:
        """The tool definition in the shape the OpenAI SDK expects.

        `[VERIFIED: openai 3.3.1, chat_completion_function_tool_param.py and
        shared_params/function_definition.py]` - {"type": "function", "function":
        {name, description, parameters}} where parameters is a JSON Schema object.
        """
        schema = self.args_model.model_json_schema()
        # `strict` mode requires additionalProperties: false, and refusing unknown
        # properties is what we want regardless of whether strict is enabled.
        schema["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolRegistry:
    """The allowlist. A model may call nothing that is not in here."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise TayrError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """Look up a tool, or raise. Never returns a fallback."""
        spec = self._tools.get(name)
        if spec is None:
            raise UnknownToolError(
                f"{name!r} is not a registered tool. Registered: {', '.join(sorted(self._tools))}"
            )
        return spec

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[n] for n in sorted(self._tools)]

    def acting_tools(self) -> set[str]:
        return {n for n, s in self._tools.items() if s.effect is ToolEffect.ACTING}

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [spec.openai_schema() for spec in self.specs()]

    def invoke(
        self, name: str, raw_arguments: str | dict[str, Any], context: Any, *, round_index: int = 0
    ) -> ToolInvocation:
        """Validate and run one tool call.

        `raw_arguments` is whatever the model produced - a JSON string from the API, or
        a dict in tests. It is never trusted: it is parsed, validated against the tool's
        schema, and only then passed to the handler.

        An unknown tool name raises `UnknownToolError` and is logged as a security
        event. Everything else - bad JSON, failed validation, a handler that raised -
        returns a failed ToolInvocation the model can see and react to.
        """
        started = time.perf_counter()

        try:
            spec = self.get(name)
        except UnknownToolError:
            log_security_event(
                SecurityEvent.AUTHZ_DENIED,
                outcome="unknown_tool",
                requested_tool=name[:64],
                registered=len(self._tools),
            )
            raise

        try:
            payload = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            if not isinstance(payload, dict):
                raise ToolArgumentError(
                    f"{name}: arguments must be a JSON object, got {type(payload).__name__}"
                )
            args = spec.args_model.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, ToolArgumentError) as exc:
            return ToolInvocation(
                tool_name=name,
                arguments=_safe_args(raw_arguments),
                ok=False,
                error=f"invalid arguments: {exc}"[:500],
                duration_ms=(time.perf_counter() - started) * 1000,
                round_index=round_index,
            )

        try:
            result = spec.handler(context, args)
        except TayrError as exc:
            # A tool failure the agent understands. It escalates as uncertain rather
            # than dismissing - see verdicts.Uncertainty.TOOL_FAILURE.
            return ToolInvocation(
                tool_name=name,
                arguments=args.model_dump(mode="json"),
                ok=False,
                error=str(exc)[:500],
                duration_ms=(time.perf_counter() - started) * 1000,
                round_index=round_index,
            )
        except Exception as exc:  # noqa: BLE001 - deliberate boundary catch, see below
            # A tool handler must not be able to crash the agent loop, so this catch is
            # broad on purpose. It is NOT a swallowed error: the failure is recorded as
            # a failed ToolInvocation, surfaced to the model, written to the audit
            # record, and resolves to ESCALATE via Uncertainty.TOOL_FAILURE. The
            # exception type is recorded; its message is not, because an arbitrary
            # exception's text can carry a path or a query.
            return ToolInvocation(
                tool_name=name,
                arguments=args.model_dump(mode="json"),
                ok=False,
                error=f"tool failed unexpectedly ({type(exc).__name__})",
                duration_ms=(time.perf_counter() - started) * 1000,
                round_index=round_index,
            )

        return ToolInvocation(
            tool_name=name,
            arguments=args.model_dump(mode="json"),
            ok=True,
            result=result,
            duration_ms=(time.perf_counter() - started) * 1000,
            round_index=round_index,
        )


def _safe_args(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Best-effort representation of arguments that failed to parse, for the record."""
    if isinstance(raw, dict):
        return raw
    return {"_unparsed": raw[:200]}
