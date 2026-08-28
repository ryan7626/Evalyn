import time
from types import SimpleNamespace

import pytest

from evalyn.llm import claude_provider as provider_module
from evalyn.llm.claude_provider import ClaudeProvider, ClaudeProviderError


class _FakeClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.options: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []
        self.messages = self
        self.closed = False

    def with_options(self, **options):
        self.options.append(options)
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def close(self) -> None:
        self.closed = True


def _response(*blocks: object) -> object:
    return SimpleNamespace(
        content=list(blocks),
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> tuple[_FakeClient, dict[str, object]]:
    client = _FakeClient(response)
    constructor: dict[str, object] = {}

    def fake_anthropic(**kwargs):
        constructor.update(kwargs)
        return client

    monkeypatch.setattr(provider_module.anthropic, "Anthropic", fake_anthropic)
    return client, constructor


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ClaudeProviderError, match="ANTHROPIC_API_KEY"):
        ClaudeProvider()


def test_pins_official_endpoint_and_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://attacker.invalid")
    _, constructor = _install_fake_client(
        monkeypatch,
        _response(SimpleNamespace(type="text", text="ok")),
    )

    ClaudeProvider(api_key="secret")

    assert constructor == {
        "api_key": "secret",
        "base_url": "https://api.anthropic.com",
        "timeout": 120.0,
        "max_retries": 0,
    }


def test_collects_all_text_blocks_with_one_total_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _install_fake_client(
        monkeypatch,
        _response(
            SimpleNamespace(type="text", text="hello "),
            SimpleNamespace(type="tool_use", text=None),
            SimpleNamespace(type="text", text="world"),
        ),
    )
    provider = ClaudeProvider(api_key="secret", timeout_seconds=30)

    assert provider.complete("system", "user") == "hello world"
    assert client.options[0]["max_retries"] == 0
    assert 0 < client.options[0]["timeout"] <= 30
    assert client.calls[0]["messages"] == [{"role": "user", "content": "user"}]


def test_rejects_response_without_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_client(
        monkeypatch,
        _response(SimpleNamespace(type="tool_use", text=None)),
    )
    provider = ClaudeProvider(api_key="secret")

    with pytest.raises(ClaudeProviderError, match="did not contain text"):
        provider.complete("system", "user")


def test_rejects_malformed_response_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_client(monkeypatch, SimpleNamespace(content=None))
    provider = ClaudeProvider(api_key="secret")

    with pytest.raises(ClaudeProviderError, match="content is malformed"):
        provider.complete("system", "user")


def test_invalid_json_does_not_echo_raw_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_client(
        monkeypatch,
        _response(SimpleNamespace(type="text", text="SENSITIVE_RAW_OUTPUT")),
    )
    provider = ClaudeProvider(api_key="secret")

    with pytest.raises(ValueError) as caught:
        provider.complete_json("system", "user")

    assert "SENSITIVE_RAW_OUTPUT" not in str(caught.value)


def test_complete_json_rejects_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_client(
        monkeypatch,
        _response(SimpleNamespace(type="text", text="1")),
    )
    provider = ClaudeProvider(api_key="secret")

    with pytest.raises(ValueError, match="object or array"):
        provider.complete_json("system", "user")


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), 601])
def test_rejects_invalid_timeout(timeout: object) -> None:
    with pytest.raises(ClaudeProviderError, match="timeout_seconds"):
        ClaudeProvider(
            api_key="secret",
            timeout_seconds=timeout,  # type: ignore[arg-type]
        )


def test_rejects_unbounded_completion_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_client(
        monkeypatch,
        _response(SimpleNamespace(type="text", text="unused")),
    )
    provider = ClaudeProvider(api_key="secret")

    with pytest.raises(ClaudeProviderError, match="max_tokens"):
        provider.complete("system", "user", max_tokens=provider_module.MAX_COMPLETION_TOKENS + 1)


def test_hard_deadline_stops_waiting_for_slow_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _install_fake_client(
        monkeypatch,
        _response(SimpleNamespace(type="text", text="unused")),
    )

    def blocked_create(**_kwargs):
        while not client.closed:
            time.sleep(0.005)
        return client.response

    client.create = blocked_create  # type: ignore[method-assign]
    provider = ClaudeProvider(api_key="secret", timeout_seconds=0.05)
    started = time.monotonic()

    with pytest.raises(ClaudeProviderError, match="timed out"):
        provider.complete("system", "user")

    assert time.monotonic() - started < 0.5
