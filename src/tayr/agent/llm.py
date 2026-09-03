"""The model interface for the agent loop.

Separate from `tayr.llm.provider`, which does single-shot summarisation. Tool calling
needs a different shape: messages in, and either tool calls or text out.

Shapes verified against the installed SDK, not written from memory
`[VERIFIED: openai 3.3.1 - chat_completion_function_tool_param.py,
chat_completion_message_function_tool_call.py, chat_completion_tool_message_param.py]`:

  request tool    {"type": "function", "function": {name, description, parameters}}
  response call   .id, .function.name, .function.arguments (a JSON **string**)
  tool result     {"role": "tool", "tool_call_id": ..., "content": ...}

The arguments field being a string the model composes is exactly why the registry
validates before dispatch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from tayr.llm.provider import LLMUnavailableError, UsageRecord

_REQUEST_TIMEOUT_SECONDS = 30.0

DEFAULT_AGENT_MODEL = "gpt-5-mini"
"""`[VERIFIED: openai 3.3.1, openai/types/shared/chat_model.py]` - present in the SDK's
ChatModel literal. **Pricing is `[UNKNOWN]`**: openai.com is egress-blocked in this
environment, so cost control is token-denominated (see `TokenBudget`), never derived
from a remembered price."""


@dataclass(frozen=True, slots=True)
class ProposedToolCall:
    """A tool call the model wants to make. Not yet validated, not yet run."""

    call_id: str
    name: str
    raw_arguments: str


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """One model response: tool calls, text, or both."""

    text: str | None = None
    tool_calls: tuple[ProposedToolCall, ...] = ()
    usage: UsageRecord = field(
        default_factory=lambda: UsageRecord(model="none", prompt_tokens=0, completion_tokens=0)
    )


@runtime_checkable
class AgentLLM(Protocol):
    """What the agent loop requires of a model."""

    @property
    def model(self) -> str: ...

    def turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_tokens: int,
    ) -> ModelTurn:
        """One round. Raises `LLMUnavailableError` on any provider failure."""
        ...


class UnavailableLLM:
    """A model that is never available.

    The default, so "no model configured" is the path every test takes unless it opts
    in - degradation is exercised constantly rather than once. Verdicts are computed by
    the rules regardless; only the prose is lost.
    """

    @property
    def model(self) -> str:
        return "none"

    def turn(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, max_tokens: int
    ) -> ModelTurn:
        del messages, tools, max_tokens
        raise LLMUnavailableError("no agent LLM is configured")


class OpenAIAgentLLM:
    """OpenAI chat completions with tool calling.

    Every failure becomes `LLMUnavailableError`, so the loop has one exception to
    degrade on rather than needing the SDK's error taxonomy.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_AGENT_MODEL,
        timeout: float = _REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._client: Any = None

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMUnavailableError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMUnavailableError(
                "the openai package is not installed; install the 'api' extra"
            ) from exc
        self._client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def turn(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, max_tokens: int
    ) -> ModelTurn:
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            # One tool per round. Parallel calls make the round cap ambiguous and make
            # the audit trail harder to read in order.
            kwargs["parallel_tool_calls"] = False

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMUnavailableError(f"OpenAI request failed: {type(exc).__name__}") from exc

        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        raw_calls = getattr(message, "tool_calls", None) or []

        usage = response.usage
        return ModelTurn(
            text=(message.content if message else None),
            tool_calls=tuple(
                ProposedToolCall(
                    call_id=c.id, name=c.function.name, raw_arguments=c.function.arguments
                )
                for c in raw_calls
                if getattr(c, "type", "function") == "function"
            ),
            usage=UsageRecord(
                model=self._model,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
        )
