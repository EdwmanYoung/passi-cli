"""TDD-style unit tests for LLM client factory and provider classes."""

from __future__ import annotations

import pytest

from passi.config import AnthropicConfig, OllamaConfig, OpenAIConfig, PassiConfig
from passi.infra.llm_client import (
    AnthropicClient,
    LLMClient,
    OllamaClient,
    OpenAIClient,
    create_llm_client,
)


class TestCreateLLMClient:
    """Factory function create_llm_client() tests."""

    def test_create_anthropic_client(self):
        cfg = PassiConfig(anthropic={"api_key": "test-key"})
        client = create_llm_client(cfg, "anthropic")
        assert isinstance(client, AnthropicClient)
        assert client.config.api_key == "test-key"

    def test_create_openai_client(self):
        cfg = PassiConfig(openai={"api_key": "sk-test"})
        client = create_llm_client(cfg, "openai")
        assert isinstance(client, OpenAIClient)
        assert client.config.api_key == "sk-test"

    def test_create_ollama_client(self):
        cfg = PassiConfig(ollama={"enabled": True, "model": "llama3.2"})
        client = create_llm_client(cfg, "ollama")
        assert isinstance(client, OllamaClient)
        assert client.config.model == "llama3.2"

    def test_create_uses_default_provider_when_none(self):
        cfg = PassiConfig(default_provider="openai", openai={"api_key": "sk-default"})
        client = create_llm_client(cfg)
        assert isinstance(client, OpenAIClient)

    def test_unknown_provider_raises_value_error(self):
        cfg = PassiConfig()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client(cfg, "gemini")

    def test_disabled_provider_raises_value_error(self):
        cfg = PassiConfig(ollama={"enabled": False})
        with pytest.raises(ValueError, match="disabled"):
            create_llm_client(cfg, "ollama")

    def test_enabled_provider_after_enable_works(self):
        cfg = PassiConfig(ollama={"enabled": True})
        client = create_llm_client(cfg, "ollama")
        assert isinstance(client, OllamaClient)


class TestAnthropicClient:
    """AnthropicClient construction and tool conversion."""

    def test_construction_stores_config(self):
        cfg = AnthropicConfig(api_key="ak-123", model="claude-opus-4-7", max_tokens=4096)
        client = AnthropicClient(cfg)
        assert client.config.api_key == "ak-123"
        assert client.config.model == "claude-opus-4-7"

    def test_supports_tool_use(self):
        client = AnthropicClient(AnthropicConfig(api_key="k"))
        assert client.supports_tool_use() is True

    def test_convert_tools_openai_to_anthropic_format(self):
        client = AnthropicClient(AnthropicConfig(api_key="k"))
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from disk.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        result = client._convert_tools(openai_tools)
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["description"] == "Read a file from disk."
        assert "input_schema" in result[0]
        assert result[0]["input_schema"]["type"] == "object"

    def test_convert_tools_passes_anthropic_format_through(self):
        client = AnthropicClient(AnthropicConfig(api_key="k"))
        already_anthropic = [
            {"name": "my_tool", "description": "d", "input_schema": {"type": "object"}}
        ]
        result = client._convert_tools(already_anthropic)
        assert result == already_anthropic

    def test_convert_messages_skips_system_role(self):
        client = AnthropicClient(AnthropicConfig(api_key="k"))
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        converted = client._convert_messages(msgs)
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_convert_messages_handles_content_blocks(self):
        client = AnthropicClient(AnthropicConfig(api_key="k"))
        msgs = [
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        converted = client._convert_messages(msgs)
        assert len(converted) == 1
        assert converted[0]["content"] == [{"type": "text", "text": "hi"}]


class TestOpenAIClient:
    """OpenAIClient construction and configuration."""

    def test_construction_with_base_url(self):
        cfg = OpenAIConfig(api_key="sk-xyz", base_url="https://api.example.com/v1")
        client = OpenAIClient(cfg)
        assert client.config.base_url == "https://api.example.com/v1"

    def test_construction_without_base_url(self):
        cfg = OpenAIConfig(api_key="sk-xyz")
        client = OpenAIClient(cfg)
        assert client.config.base_url is None

    def test_supports_tool_use(self):
        client = OpenAIClient(OpenAIConfig(api_key="k"))
        assert client.supports_tool_use() is True


class TestOllamaClient:
    """OllamaClient wraps OpenAIClient with different defaults."""

    def test_is_openai_client_subclass(self):
        client = OllamaClient(OllamaConfig(enabled=True))
        assert isinstance(client, OpenAIClient)

    def test_uses_ollama_default_base_url(self):
        cfg = OllamaConfig(enabled=True)
        client = OllamaClient(cfg)
        assert "11434" in client.config.base_url

    def test_custom_base_url_preserved(self):
        cfg = OllamaConfig(enabled=True, base_url="http://192.168.1.100:9999/v1")
        client = OllamaClient(cfg)
        assert client.config.base_url == "http://192.168.1.100:9999/v1"

    def test_supports_tool_use(self):
        client = OllamaClient(OllamaConfig(enabled=True))
        assert client.supports_tool_use() is True


class TestLLMClientABC:
    """LLMClient abstract base class contract."""

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            LLMClient(AnthropicConfig(api_key="k"))  # type: ignore[abstract]
