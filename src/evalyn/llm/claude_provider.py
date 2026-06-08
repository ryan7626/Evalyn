"""Claude (Anthropic) LLM provider for Evalyn.

Implements the BaseLLM interface using the Anthropic Python SDK.
Reads ANTHROPIC_API_KEY from environment. Includes retry with
exponential backoff for transient API errors.
"""

import json
import logging
import os
import time
from typing import Optional

import anthropic

from evalyn.llm.base import BaseLLM

logger = logging.getLogger("evalyn.llm.claude")

# Retry settings
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
BACKOFF_MULTIPLIER = 2.0


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
        api_key: Optional[str] = None,
    ):
        self.model = model
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ClaudeProviderError(
                "ANTHROPIC_API_KEY not set. "
                "Set it as an environment variable before running Evalyn."
            )
        self.client = anthropic.Anthropic(api_key=resolved_key)

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
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Claude response is not valid JSON: {e}\n"
                f"Raw response (first 500 chars): {raw[:500]}"
            )

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
    ) -> str:
        """Make an API call with retry and exponential backoff."""
        backoff = INITIAL_BACKOFF_SECONDS

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug(
                    f"Claude API call (attempt {attempt}/{MAX_RETRIES}): "
                    f"model={self.model}, max_tokens={max_tokens}"
                )

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                text = response.content[0].text

                logger.info(
                    f"Claude response: "
                    f"input_tokens={response.usage.input_tokens}, "
                    f"output_tokens={response.usage.output_tokens}"
                )

                return text

            except anthropic.RateLimitError:
                if attempt == MAX_RETRIES:
                    raise ClaudeProviderError(
                        f"Rate limited after {MAX_RETRIES} retries."
                    )
                logger.warning(
                    f"Rate limited, retrying in {backoff}s "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER

            except anthropic.APIConnectionError as e:
                if attempt == MAX_RETRIES:
                    raise ClaudeProviderError(
                        f"API connection failed after {MAX_RETRIES} retries: {e}"
                    )
                logger.warning(
                    f"Connection error, retrying in {backoff}s "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER

            except anthropic.APIStatusError as e:
                # Non-retryable API errors (auth, bad request, etc.)
                raise ClaudeProviderError(
                    f"Claude API error ({e.status_code}): {e.message}"
                )

        # Should not reach here, but just in case
        raise ClaudeProviderError("Unexpected: exhausted retries without result.")
