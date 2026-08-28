"""Claude (Anthropic) LLM provider for Evalyn.

Implements the BaseLLM interface using the Anthropic Python SDK.
Reads ANTHROPIC_API_KEY from environment. Includes retry with
exponential backoff for transient API errors.
"""

import json
import logging
import math
import os
import time

import anthropic

from evalyn.llm.base import BaseLLM
from evalyn.llm.deadline import run_with_hard_deadline

logger = logging.getLogger("evalyn.llm.claude")

# Retry settings
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
BACKOFF_MULTIPLIER = 2.0
ANTHROPIC_API_URL = "https://api.anthropic.com"
MAX_COMPLETION_TOKENS = 64_000
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class ClaudeProviderError(Exception):
    """Raised when the Claude provider encounters a fatal error."""

    pass


class ClaudeProvider(BaseLLM):
    """Claude/Anthropic implementation of BaseLLM.

    Args:
        model: Model identifier (default: claude-sonnet-4-20250514).
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        timeout_seconds: float = 120,
    ):
        if not isinstance(model, str) or not model.strip() or len(model) > 256:
            raise ClaudeProviderError("model must be a nonempty string")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > 600
        ):
            raise ClaudeProviderError("timeout_seconds must be greater than zero and at most 600")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ClaudeProviderError(
                "ANTHROPIC_API_KEY not set. "
                "Set it as an environment variable before running Evalyn."
            )
        # Pin the official endpoint instead of allowing the SDK to honor an
        # ambient ANTHROPIC_BASE_URL and forward this credential elsewhere.
        self.client = anthropic.Anthropic(
            api_key=resolved_key,
            base_url=ANTHROPIC_API_URL,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text completion via Claude."""
        return self._call(system_prompt, user_prompt, max_tokens=max_tokens)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> dict | list:
        """Generate a structured JSON completion via Claude.

        Appends JSON instruction to system prompt and parses the response.
        """
        json_system = (
            f"{system_prompt}\n\n"
            "IMPORTANT: Respond ONLY with valid JSON. No markdown fences, "
            "no extra text. Your entire response must be parseable as JSON."
        )
        raw = self._call(json_system, user_prompt, max_tokens=max_tokens)

        # Strip markdown fences if the model adds them despite instructions
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json) and last line (```)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Claude response is not valid JSON: {e}") from e
        if not isinstance(parsed, (dict, list)):
            raise ValueError("Claude response JSON must be an object or array")
        return parsed

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
    ) -> str:
        """Make an API call with retry and exponential backoff."""
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
            or max_tokens > MAX_COMPLETION_TOKENS
        ):
            raise ClaudeProviderError(f"max_tokens must be between 1 and {MAX_COMPLETION_TOKENS}")
        request_size = len(system_prompt.encode("utf-8")) + len(user_prompt.encode("utf-8"))
        if request_size > MAX_REQUEST_BYTES:
            raise ClaudeProviderError(f"Claude request exceeds {MAX_REQUEST_BYTES} bytes")
        return run_with_hard_deadline(
            lambda: self._call_with_retries(
                system_prompt,
                user_prompt,
                max_tokens=max_tokens,
            ),
            timeout_seconds=self.timeout_seconds,
            timeout_error=lambda: ClaudeProviderError("Claude request timed out"),
            cancel=self.client.close,
        )

    def _call_with_retries(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
    ) -> str:
        backoff = INITIAL_BACKOFF_SECONDS
        deadline = time.monotonic() + self.timeout_seconds

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug(
                    f"Claude API call (attempt {attempt}/{MAX_RETRIES}): "
                    f"model={self.model}, max_tokens={max_tokens}"
                )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ClaudeProviderError("Claude request timed out")
                response = self.client.with_options(
                    timeout=remaining,
                    max_retries=0,
                ).messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                if time.monotonic() >= deadline:
                    raise ClaudeProviderError("Claude request timed out")

                content = getattr(response, "content", None)
                if not isinstance(content, (list, tuple)):
                    raise ClaudeProviderError("Claude response content is malformed")
                text_parts = [
                    block.text
                    for block in content
                    if getattr(block, "type", None) == "text"
                    and isinstance(getattr(block, "text", None), str)
                ]
                if not text_parts:
                    raise ClaudeProviderError("Claude response did not contain text")
                text = "".join(text_parts)
                if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
                    raise ClaudeProviderError(f"Claude response exceeds {MAX_RESPONSE_BYTES} bytes")

                usage = getattr(response, "usage", None)
                logger.info(
                    "Claude response: input_tokens=%s, output_tokens=%s",
                    getattr(usage, "input_tokens", "unknown"),
                    getattr(usage, "output_tokens", "unknown"),
                )

                return text

            except anthropic.RateLimitError as exc:
                if attempt == MAX_RETRIES:
                    raise ClaudeProviderError(f"Rate limited after {MAX_RETRIES} retries.") from exc
                logger.warning(
                    f"Rate limited, retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})"
                )
                _sleep_before_retry(deadline, backoff)
                backoff *= BACKOFF_MULTIPLIER

            except anthropic.APIConnectionError as e:
                if attempt == MAX_RETRIES:
                    raise ClaudeProviderError(
                        f"API connection failed after {MAX_RETRIES} retries"
                    ) from e
                logger.warning(
                    f"Connection error, retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})"
                )
                _sleep_before_retry(deadline, backoff)
                backoff *= BACKOFF_MULTIPLIER

            except anthropic.APIStatusError as e:
                if e.status_code >= 500 and attempt < MAX_RETRIES:
                    logger.warning(
                        f"Server error, retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})"
                    )
                    _sleep_before_retry(deadline, backoff)
                    backoff *= BACKOFF_MULTIPLIER
                    continue
                raise ClaudeProviderError(f"Claude API error ({e.status_code})") from e

        # Should not reach here, but just in case
        raise ClaudeProviderError("Unexpected: exhausted retries without result.")


def _sleep_before_retry(deadline: float, delay: float) -> None:
    if deadline - time.monotonic() <= delay:
        raise ClaudeProviderError("Claude request timed out")
    time.sleep(delay)
