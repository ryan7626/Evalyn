import json
import time
import urllib.request
from io import BytesIO
from threading import Event

import pytest

from evalyn.llm import openai_compatible_provider as provider_module
from evalyn.llm.openai_compatible_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleProviderError,
)


class _Response:
    def __init__(self, body: object, *, headers: dict[str, str] | None = None) -> None:
        encoded = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.body = BytesIO(encoded)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body.read(size)


def test_remote_endpoint_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAICompatibleProviderError, match="OPENAI_API_KEY"):
        OpenAICompatibleProvider("model", base_url="https://models.example/v1")


def test_local_endpoint_can_run_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen: list[urllib.request.Request] = []

    def fake_open(request, *, timeout):
        seen.append(request)
        assert 0 < timeout <= 120
        return _Response({"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr(provider_module, "_open_no_redirect", fake_open)
    provider = OpenAICompatibleProvider("local-model", base_url="http://localhost:11434/v1")

    assert provider.complete("system", "user") == "hello"
    assert seen[0].full_url == "http://localhost:11434/v1/chat/completions"
    assert seen[0].get_header("Authorization") is None


def test_complete_json_parses_fenced_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provider_module,
        "_open_no_redirect",
        lambda *_args, **_kwargs: _Response(
            {"choices": [{"message": {"content": '```json\n{"ok": true}\n```'}}]}
        ),
    )
    provider = OpenAICompatibleProvider("local-model", base_url="http://127.0.0.1:11434/v1")

    assert provider.complete_json("system", "user") == {"ok": True}


def test_rejects_non_http_base_url() -> None:
    with pytest.raises(OpenAICompatibleProviderError, match="HTTP"):
        OpenAICompatibleProvider("model", base_url="file:///tmp/model", api_key="key")


def test_rejects_remote_plaintext_endpoint() -> None:
    with pytest.raises(OpenAICompatibleProviderError, match="must use HTTPS"):
        OpenAICompatibleProvider("model", base_url="http://models.example/v1", api_key="key")


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:secret@models.example/v1",
        "https://models.example/v1?token=secret",
        "https://models.example/v1#fragment",
    ],
)
def test_rejects_credentials_or_suffix_data_in_base_url(base_url: str) -> None:
    with pytest.raises(OpenAICompatibleProviderError, match="cannot contain"):
        OpenAICompatibleProvider("model", base_url=base_url, api_key="key")


def test_redirects_are_refused_before_a_new_request_is_created() -> None:
    handler = provider_module._RejectRedirects()
    request = urllib.request.Request(
        "https://models.example/v1/chat/completions",
        headers={"Authorization": "Bearer secret"},
    )

    with pytest.raises(OpenAICompatibleProviderError, match="redirects are refused"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://169.254.169.254/latest/meta-data",
        )


def test_rejects_oversized_success_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provider_module,
        "_open_no_redirect",
        lambda *_args, **_kwargs: _Response(
            b"{}",
            headers={"Content-Length": str(provider_module._MAX_RESPONSE_BYTES + 1)},
        ),
    )
    provider = OpenAICompatibleProvider(
        "model", base_url="https://models.example/v1", api_key="key"
    )

    with pytest.raises(OpenAICompatibleProviderError, match="response exceeds"):
        provider.complete("system", "user")


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
    ],
)
def test_malformed_success_response_raises_provider_error(
    monkeypatch: pytest.MonkeyPatch,
    body: object,
) -> None:
    monkeypatch.setattr(
        provider_module,
        "_open_no_redirect",
        lambda *_args, **_kwargs: _Response(body),
    )
    provider = OpenAICompatibleProvider(
        "model", base_url="https://models.example/v1", api_key="key"
    )

    with pytest.raises(OpenAICompatibleProviderError):
        provider.complete("system", "user")


def test_complete_json_rejects_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provider_module,
        "_open_no_redirect",
        lambda *_args, **_kwargs: _Response({"choices": [{"message": {"content": "1"}}]}),
    )
    provider = OpenAICompatibleProvider(
        "model", base_url="https://models.example/v1", api_key="key"
    )

    with pytest.raises(ValueError, match="object or array"):
        provider.complete_json("system", "user")


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), 601])
def test_rejects_invalid_timeout(timeout: object) -> None:
    with pytest.raises(OpenAICompatibleProviderError, match="timeout_seconds"):
        OpenAICompatibleProvider(
            "model",
            base_url="https://models.example/v1",
            api_key="key",
            timeout_seconds=timeout,  # type: ignore[arg-type]
        )


def test_hard_deadline_closes_a_slow_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowResponse:
        headers: dict[str, str] = {}

        def __init__(self) -> None:
            self.closed = Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            self.closed.wait(5)
            return b""

        def close(self) -> None:
            self.closed.set()

    response = SlowResponse()
    monkeypatch.setattr(
        provider_module,
        "_open_no_redirect",
        lambda *_args, **_kwargs: response,
    )
    provider = OpenAICompatibleProvider(
        "model",
        base_url="https://models.example/v1",
        api_key="key",
        timeout_seconds=0.05,
    )
    started = time.monotonic()

    with pytest.raises(OpenAICompatibleProviderError, match="timed out"):
        provider.complete("system", "user")

    assert time.monotonic() - started < 0.5


def test_late_response_registration_observes_cancellation() -> None:
    closer = provider_module._ResponseCloser()
    closed = Event()
    response = type("Response", (), {"close": lambda _self: closed.set()})()

    closer.close()
    closer.register(response)

    assert closed.is_set()
