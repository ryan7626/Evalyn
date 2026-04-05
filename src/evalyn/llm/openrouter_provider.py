"""OpenRouter LLM provider for Evalyn.

Implements the BaseLLM interface using raw HTTP to OpenRouter's
OpenAI-compatible API format. Reads OPENROUTER_API_KEY from environment.
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Optional

from evalyn.llm.base import BaseLLM

logger = logging.getLogger("evalyn.llm.openrouter")

# Retry settings
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
BACKOFF_MULTIPLIER = 2.0


class OpenRouterProviderError(Exception):
    """Raised when the OpenRouter provider encounters a fatal error."""
    pass


class OpenRouterProvider(BaseLLM):
    """OpenRouter implementation of BaseLLM.

    Args:
        model: Model identifier (default: anthropic/claude-3.5-sonnet).
        api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterProviderError(
                "OPENROUTER_API_KEY not set. "
                "Set it as an environment variable before running Evalyn."
            )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text completion via OpenRouter."""
        return self._call(system_prompt, user_prompt, max_tokens=max_tokens)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> dict:
        """Generate a structured JSON completion via OpenRouter.

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
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"OpenRouter response is not valid JSON: {e}\n"
                f"Raw response (first 500 chars): {raw[:500]}"
            )

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
    ) -> str:
        """Make an API call with retry and exponential backoff using urllib."""
        backoff = INITIAL_BACKOFF_SECONDS
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/yourusername/Evalyn",
            "X-Title": "Evalyn Local Run",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(data).encode("utf-8"), 
            headers=headers,
            method="POST"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug(
                    f"OpenRouter API call (attempt {attempt}/{MAX_RETRIES}): "
                    f"model={self.model}, max_tokens={max_tokens}"
                )

                with urllib.request.urlopen(req, timeout=120) as response:
                    body = response.read().decode("utf-8")
                    result = json.loads(body)
                    
                    text = result["choices"][0]["message"]["content"]

                    usage = result.get("usage", {})
                    logger.info(
                        f"OpenRouter response: "
                        f"input_tokens={usage.get('prompt_tokens', 0)}, "
                        f"output_tokens={usage.get('completion_tokens', 0)}"
                    )

                    return text

            except urllib.error.HTTPError as e:
                # 429 Too Many Requests
                if e.code == 429:
                    if attempt == MAX_RETRIES:
                        raise OpenRouterProviderError(f"Rate limited after {MAX_RETRIES} retries.")
                    
                    logger.warning(
                        f"Rate limited (429), retrying in {backoff}s "
                        f"(attempt {attempt}/{MAX_RETRIES})"
                    )
                    time.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER
                elif e.code >= 500:
                    if attempt == MAX_RETRIES:
                        raise OpenRouterProviderError(f"Server error {e.code} after {MAX_RETRIES} retries: {e.read().decode('utf-8')}")
                    
                    logger.warning(
                        f"Server error {e.code}, retrying in {backoff}s "
                        f"(attempt {attempt}/{MAX_RETRIES})"
                    )
                    time.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER
                else:
                    raise OpenRouterProviderError(f"OpenRouter API error ({e.code}): {e.read().decode('utf-8')}")
                    
            except urllib.error.URLError as e:
                if attempt == MAX_RETRIES:
                    raise OpenRouterProviderError(f"API connection failed after {MAX_RETRIES} retries: {e.reason}")
                logger.warning(
                    f"Connection error, retrying in {backoff}s "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER

            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise OpenRouterProviderError(f"Unexpected error after {MAX_RETRIES} retries: {e}")
                logger.warning(f"Unexpected error: {e}, retrying in {backoff}s")
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER

        # Should not reach here, but just in case
        raise OpenRouterProviderError("Unexpected: exhausted retries without result.")
