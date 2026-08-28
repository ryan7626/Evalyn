"""Provider for OpenAI-compatible chat-completions endpoints."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import suppress
from threading import Lock
from typing import BinaryIO

from evalyn.llm.base import BaseLLM
from evalyn.llm.deadline import run_with_hard_deadline


class OpenAICompatibleProviderError(RuntimeError):
    pass


_MAX_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_COMPLETION_TOKENS = 64_000
_READ_CHUNK_BYTES = 64 * 1024


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Never forward a provider credential to a redirected origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise OpenAICompatibleProviderError(
            "Compatible endpoint redirects are refused; configure the final API URL"
        )


def _open_no_redirect(request: urllib.request.Request, *, timeout: float):
    opener = urllib.request.build_opener(_RejectRedirects())
    return opener.open(request, timeout=timeout)


class _ResponseCloser:
    def __init__(self) -> None:
        self._lock = Lock()
        self._response: object | None = None
        self._cancelled = False

    def register(self, response: object) -> None:
        with self._lock:
            cancelled = self._cancelled
            if not cancelled:
                self._response = response
        if cancelled:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def unregister(self, response: object) -> None:
        with self._lock:
            if self._response is response:
                self._response = None

    def close(self) -> None:
        with self._lock:
            self._cancelled = True
            response = self._response
        close = getattr(response, "close", None)
        if callable(close):
            close()


class OpenAICompatibleProvider(BaseLLM):
    """Use a hosted or local OpenAI-compatible ``/chat/completions`` API."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 120,
    ) -> None:
        if not isinstance(model, str) or not model.strip() or len(model) > 256:
            raise OpenAICompatibleProviderError("model must be a nonempty string")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > 600
        ):
            raise OpenAICompatibleProviderError(
                "timeout_seconds must be greater than zero and at most 600"
            )
        self.model = model
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.timeout_seconds = float(timeout_seconds)
        if len(self.base_url) > 2048:
            raise OpenAICompatibleProviderError("OPENAI_BASE_URL is too long")
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise OpenAICompatibleProviderError("OPENAI_BASE_URL must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise OpenAICompatibleProviderError(
                "OPENAI_BASE_URL cannot contain credentials, a query, or a fragment"
            )
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not local:
            raise OpenAICompatibleProviderError(
                "Non-local OPENAI_BASE_URL endpoints must use HTTPS"
            )
        if not self.api_key and not local:
            raise OpenAICompatibleProviderError(
                "OPENAI_API_KEY is required for non-local compatible endpoints"
            )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> str:
        return self._call(system_prompt, user_prompt, max_tokens)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> dict | list:
        raw = self._call(
            system_prompt + "\n\nReturn only valid JSON without markdown fences.",
            user_prompt,
            max_tokens,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines.pop()
            cleaned = "\n".join(lines)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Compatible endpoint returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, (dict, list)):
            raise ValueError("Compatible endpoint JSON must be an object or array")
        return parsed

    def _call(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
            or max_tokens > _MAX_COMPLETION_TOKENS
        ):
            raise OpenAICompatibleProviderError(
                f"max_tokens must be between 1 and {_MAX_COMPLETION_TOKENS}"
            )
        payload = json.dumps(
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        if len(payload) > _MAX_REQUEST_BYTES:
            raise OpenAICompatibleProviderError(
                f"Compatible endpoint request exceeds {_MAX_REQUEST_BYTES} bytes"
            )
        headers = {"Content-Type": "application/json", "User-Agent": "evalyn/0.2"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload, headers=headers, method="POST"
        )
        response_closer = _ResponseCloser()
        return run_with_hard_deadline(
            lambda: self._perform_request(request, response_closer),
            timeout_seconds=self.timeout_seconds,
            timeout_error=lambda: OpenAICompatibleProviderError(
                "Compatible endpoint request timed out"
            ),
            cancel=response_closer.close,
        )

    def _perform_request(
        self,
        request: urllib.request.Request,
        response_closer: _ResponseCloser,
    ) -> str:
        delay = 1.0
        deadline = time.monotonic() + self.timeout_seconds
        for attempt in range(3):
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OpenAICompatibleProviderError("Compatible endpoint request timed out")
                response = _open_no_redirect(request, timeout=remaining)
                response_closer.register(response)
                try:
                    with response:
                        body = _read_bounded_response(response, deadline=deadline)
                finally:
                    response_closer.unregister(response)
                if time.monotonic() >= deadline:
                    raise OpenAICompatibleProviderError("Compatible endpoint request timed out")
                try:
                    result = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise OpenAICompatibleProviderError(
                        "Compatible endpoint returned malformed JSON"
                    ) from exc
                content = _extract_content(result)
                if not isinstance(content, str):
                    raise OpenAICompatibleProviderError("Endpoint returned non-text content")
                return content
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                with suppress(OSError):
                    exc.close()
                if not retryable or attempt == 2:
                    raise OpenAICompatibleProviderError(
                        f"Compatible endpoint error {exc.code}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 2:
                    raise OpenAICompatibleProviderError(
                        "Compatible endpoint connection failed"
                    ) from exc
            remaining = deadline - time.monotonic()
            if remaining <= delay:
                raise OpenAICompatibleProviderError("Compatible endpoint request timed out")
            time.sleep(delay)
            delay *= 2
        raise OpenAICompatibleProviderError("Compatible endpoint request failed")


def _read_bounded_response(response: BinaryIO, *, deadline: float) -> bytes:
    headers = getattr(response, "headers", None)
    declared = headers.get("Content-Length") if headers is not None else None
    if declared is not None:
        try:
            if int(declared) > _MAX_RESPONSE_BYTES:
                raise OpenAICompatibleProviderError(
                    f"Compatible endpoint response exceeds {_MAX_RESPONSE_BYTES} bytes"
                )
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    reader = getattr(response, "read1", None) or response.read
    while True:
        if time.monotonic() >= deadline:
            raise OpenAICompatibleProviderError("Compatible endpoint response timed out")
        chunk = reader(min(_READ_CHUNK_BYTES, _MAX_RESPONSE_BYTES + 1 - total))
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise OpenAICompatibleProviderError("Compatible endpoint returned invalid bytes")
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_RESPONSE_BYTES:
            raise OpenAICompatibleProviderError(
                f"Compatible endpoint response exceeds {_MAX_RESPONSE_BYTES} bytes"
            )
    return b"".join(chunks)


def _extract_content(result: object) -> str:
    if not isinstance(result, dict):
        raise OpenAICompatibleProviderError("Compatible endpoint response must be an object")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise OpenAICompatibleProviderError("Compatible endpoint response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise OpenAICompatibleProviderError("Compatible endpoint response has no message content")
    content = message["content"]
    if not isinstance(content, str):
        raise OpenAICompatibleProviderError("Endpoint returned non-text content")
    return content
