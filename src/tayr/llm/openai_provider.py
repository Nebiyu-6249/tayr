"""OpenAI provider.

Model ids are `[VERIFIED]` from the SDK's generated
`openai/types/shared/chat_model.py` (openai 3.3.1). **Pricing is `[UNKNOWN]`** — the
pricing page was egress-blocked during Phase 0 and has not been checked since. Do not
set a dollar-denominated cost cap from a remembered price; the token cap in
`CostTracker` is the control that actually holds.

The API key is read from the environment, server-side only. It must never reach a
frontend bundle, a client request, or a log — `security.audit` scrubs `api_key` at any
nesting depth.
"""

from __future__ import annotations

import os

from tayr.llm.provider import LLMUnavailableError, UsageRecord

# Verified to exist in the SDK's ChatModel literal. Small models are the default: a
# summary of a structured record is not a task that needs a frontier model.
DEFAULT_MODEL = "gpt-5-mini"

_REQUEST_TIMEOUT_SECONDS = 30.0


class OpenAIProvider:
    """Calls OpenAI's chat completions API.

    Every failure becomes `LLMUnavailableError`, so callers have exactly one exception to
    degrade on rather than needing to know the SDK's error taxonomy.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = _REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._client: object | None = None

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self) -> object:
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

    def complete(self, *, system: str, user: str, max_tokens: int) -> tuple[str, UsageRecord]:
        client = self._ensure_client()
        try:
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                messages=[
                    # Instructions and data are separate messages. This separation is
                    # the prompt-injection boundary, not a formatting preference.
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            # Includes auth failures, rate limits, timeouts and network errors. The
            # message is not surfaced to users: it can echo request content.
            raise LLMUnavailableError(f"OpenAI request failed: {type(exc).__name__}") from exc

        choice = response.choices[0] if response.choices else None
        content = (choice.message.content if choice and choice.message else None) or ""
        usage = response.usage
        return content, UsageRecord(
            model=self._model,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
