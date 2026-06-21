"""
Unit tests for LLM providers.
All external SDK calls are mocked — no API keys or network needed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm.base import Message, calculate_cost, LLMResponse
from app.services.llm.factory import get_llm_provider, _build_provider


# ── Cost calculation ──────────────────────────────────────────────────────────

def test_calculate_cost_known_model() -> None:
    cost = calculate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    assert cost > 0
    expected = (1000 / 1000 * 0.00015) + (500 / 1000 * 0.0006)
    assert abs(cost - expected) < 1e-9


def test_calculate_cost_unknown_model_returns_zero() -> None:
    assert calculate_cost("unknown-model-xyz", 1000, 500) == 0.0


def test_calculate_cost_zero_tokens() -> None:
    assert calculate_cost("gpt-4o-mini", 0, 0) == 0.0


def test_calculate_cost_anthropic_model() -> None:
    cost = calculate_cost("claude-sonnet-4-6", 2000, 800)
    expected = (2000 / 1000 * 0.003) + (800 / 1000 * 0.015)
    assert abs(cost - expected) < 1e-9


# ── Message ───────────────────────────────────────────────────────────────────

def test_message_construction() -> None:
    m = Message(role="user", content="Hello")
    assert m.role == "user"
    assert m.content == "Hello"


# ── OpenAI adapter ────────────────────────────────────────────────────────────

@pytest.fixture
def openai_adapter():
    with patch("app.services.llm.openai_adapter.get_settings") as ms:
        ms.return_value.openai_api_key = "sk-test"
        ms.return_value.openai_default_model = "gpt-4o-mini"
        with patch("app.services.llm.openai_adapter.AsyncOpenAI"):
            from app.services.llm.openai_adapter import OpenAIAdapter
            adapter = OpenAIAdapter(model="gpt-4o-mini", api_key="sk-test")
    return adapter


@pytest.mark.asyncio
async def test_openai_complete(openai_adapter) -> None:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "Hello from GPT"
    mock_resp.choices[0].finish_reason = "stop"
    mock_resp.usage.prompt_tokens = 10
    mock_resp.usage.completion_tokens = 5
    openai_adapter._client = AsyncMock()
    openai_adapter._client.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await openai_adapter.complete([Message(role="user", content="Hi")])
    assert result.content == "Hello from GPT"
    assert result.provider == "openai"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15
    assert result.cost_usd >= 0


@pytest.mark.asyncio
async def test_openai_complete_raises_llm_error_on_failure(openai_adapter) -> None:
    from app.core.exceptions import LLMError
    openai_adapter._client = AsyncMock()
    openai_adapter._client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))
    with pytest.raises(LLMError, match="OpenAI completion failed"):
        await openai_adapter.complete([Message(role="user", content="Hi")])


# ── Anthropic adapter ─────────────────────────────────────────────────────────

@pytest.fixture
def anthropic_adapter():
    with patch("app.services.llm.anthropic_adapter.get_settings") as ms:
        ms.return_value.anthropic_api_key = "ant-test"
        ms.return_value.anthropic_default_model = "claude-sonnet-4-6"
        with patch("app.services.llm.anthropic_adapter.anthropic"):
            from app.services.llm.anthropic_adapter import AnthropicAdapter
            adapter = AnthropicAdapter(model="claude-sonnet-4-6", api_key="ant-test")
    return adapter


@pytest.mark.asyncio
async def test_anthropic_complete(anthropic_adapter) -> None:
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="Hello from Claude")]
    mock_resp.stop_reason = "end_turn"
    mock_resp.usage.input_tokens = 20
    mock_resp.usage.output_tokens = 8
    anthropic_adapter._client = AsyncMock()
    anthropic_adapter._client.messages.create = AsyncMock(return_value=mock_resp)

    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hi Claude"),
    ]
    result = await anthropic_adapter.complete(messages)
    assert result.content == "Hello from Claude"
    assert result.provider == "anthropic"
    assert result.prompt_tokens == 20
    assert result.completion_tokens == 8


def test_anthropic_split_messages(anthropic_adapter) -> None:
    from app.services.llm.anthropic_adapter import AnthropicAdapter
    messages = [
        Message(role="system", content="Be helpful."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi!"),
    ]
    system, chat = AnthropicAdapter._split_messages(messages)
    assert system == "Be helpful."
    assert len(chat) == 2
    assert chat[0]["role"] == "user"
    assert chat[1]["role"] == "assistant"


# ── Ollama adapter ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ollama_complete() -> None:
    with patch("app.services.llm.ollama_adapter.get_settings") as ms:
        ms.return_value.ollama_default_model = "llama3.2"
        ms.return_value.ollama_base_url = "http://localhost:11434"
        from app.services.llm.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="llama3.2")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"role": "assistant", "content": "Hello from Ollama"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 15,
        "eval_count": 6,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("app.services.llm.ollama_adapter.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await adapter.complete([Message(role="user", content="Hi")])

    assert result.content == "Hello from Ollama"
    assert result.provider == "ollama"
    assert result.cost_usd == 0.0
    assert result.prompt_tokens == 15


# ── Factory ───────────────────────────────────────────────────────────────────

def test_build_provider_unknown_raises() -> None:
    from app.core.exceptions import LLMError
    with pytest.raises(LLMError, match="Unknown LLM provider"):
        _build_provider("unknown-provider-xyz")


def test_get_llm_provider_explicit_openai() -> None:
    with patch("app.services.llm.factory.get_settings") as ms, \
         patch("app.services.llm.openai_adapter.get_settings") as ms2, \
         patch("app.services.llm.openai_adapter.AsyncOpenAI"):
        ms.return_value.openai_api_key = "sk-test"
        ms2.return_value.openai_api_key = "sk-test"
        ms2.return_value.openai_default_model = "gpt-4o-mini"
        provider = get_llm_provider(provider="openai", model="gpt-4o-mini")
    assert provider.provider_name == "openai"


def test_llm_response_total_cost_property() -> None:
    resp = LLMResponse(
        content="test",
        model="gpt-4o-mini",
        provider="openai",
        cost_usd=0.0025,
    )
    assert resp.total_cost == 0.0025
