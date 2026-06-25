"""TDD-style unit tests for LLM client factory and provider classes."""

from __future__ import annotations

from unittest.mock import MagicMock

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

    def test_construction_with_base_url(self):
        """AnthropicClient passes base_url to AsyncAnthropic SDK."""
        import anthropic

        # Verify that AsyncAnthropic receives base_url when configured
        original = anthropic.AsyncAnthropic
        received_kwargs: dict = {}

        class _SpyAsyncAnthropic:
            def __init__(self, **kw: Any) -> None:
                nonlocal received_kwargs
                received_kwargs = kw

        try:
            anthropic.AsyncAnthropic = _SpyAsyncAnthropic  # type: ignore[assignment]
            AnthropicClient(AnthropicConfig(
                api_key="sk-test",
                base_url="https://api.deepseek.com/anthropic",
            ))
        finally:
            anthropic.AsyncAnthropic = original

        assert received_kwargs["api_key"] == "sk-test"
        assert received_kwargs["base_url"] == "https://api.deepseek.com/anthropic"

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


# ═══════════════════════════════════════════════════════════════════
# Mock helpers for Anthropic / OpenAI response objects
# ═══════════════════════════════════════════════════════════════════


class _MockAnthropicTextBlock:
    """Simulates anthropic.types.TextBlock."""

    def __init__(self, text: str = "Hello.") -> None:
        self.type = "text"
        self.text = text


class _MockAnthropicToolUseBlock:
    """Simulates anthropic.types.ToolUseBlock."""

    def __init__(self, id: str = "tool_01", name: str = "read_file", input: dict | None = None) -> None:
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input or {}


class _MockAnthropicUsage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MockAnthropicResponse:
    def __init__(
        self,
        content: list | None = None,
        usage: _MockAnthropicUsage | None = None,
        stop_reason: str = "end_turn",
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.content = content or [_MockAnthropicTextBlock()]
        self.usage = usage or _MockAnthropicUsage()
        self.stop_reason = stop_reason
        self.model = model


def _make_anthropic_client(**kwargs: Any) -> AnthropicClient:
    """Create an AnthropicClient with a mocked SDK, skipping real API init."""
    cfg = AnthropicConfig(api_key=kwargs.pop("api_key", "sk-ant-test"), **kwargs)
    client = AnthropicClient.__new__(AnthropicClient)
    LLMClient.__init__(client, cfg)
    client._model = cfg.model
    client._client = MagicMock()
    return client


# ═══════════════════════════════════════════════════════════════════
# AnthropicClient.chat() tests
# ═══════════════════════════════════════════════════════════════════


class TestAnthropicClientChat:
    """AnthropicClient.chat() — mocked API responses."""

    @pytest.mark.asyncio
    async def test_text_only_response(self):
        """chat() with a plain text response returns correct content block."""
        client = _make_anthropic_client()
        mock_resp = _MockAnthropicResponse(
            content=[_MockAnthropicTextBlock("Here is your analysis.")],
        )

        async def _fake_create(**kw: Any) -> Any:
            return mock_resp

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        result = await client.chat(messages=[{"role": "user", "content": "Hello"}])
        assert result["content"] == [{"type": "text", "text": "Here is your analysis."}]
        assert result["tool_calls"] is None
        assert result["finish_reason"] == "end_turn"
        assert result["model"] == "claude-sonnet-4-6"
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_response_with_tool_use_blocks(self):
        """chat() with tool_use blocks returns tool_calls correctly."""
        client = _make_anthropic_client()
        mock_resp = _MockAnthropicResponse(
            content=[
                _MockAnthropicTextBlock("Let me read that file."),
                _MockAnthropicToolUseBlock(
                    id="toolu_001", name="read_file", input={"path": "/data.csv"}
                ),
            ],
            stop_reason="tool_use",
        )

        async def _fake_create(**kw: Any) -> Any:
            return mock_resp

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        result = await client.chat(messages=[{"role": "user", "content": "Read data.csv"}])
        assert len(result["content"]) == 2
        # tool_calls list populated
        assert result["tool_calls"] == [
            {"id": "toolu_001", "name": "read_file", "input": {"path": "/data.csv"}},
        ]
        assert result["finish_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_system_prompt_injected(self):
        """System prompt is passed as top-level 'system' kwarg."""
        client = _make_anthropic_client()
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockAnthropicResponse()

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            system="You are a bioinformatics expert.",
        )
        assert delivered_kwargs.get("system") == "You are a bioinformatics expert."

    @pytest.mark.asyncio
    async def test_custom_temperature_and_max_tokens(self):
        """Temperature and max_tokens overrides are forwarded."""
        client = _make_anthropic_client()
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockAnthropicResponse()

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=16000,
        )
        assert delivered_kwargs["temperature"] == 0.7
        assert delivered_kwargs["max_tokens"] == 16000

    @pytest.mark.asyncio
    async def test_tools_converted_and_passed(self):
        """OpenAI-format tools are converted to Anthropic format."""
        client = _make_anthropic_client()
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockAnthropicResponse()

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        openai_tools = [{
            "type": "function",
            "function": {
                "name": "qc_report",
                "description": "Run QC report on data.",
                "parameters": {
                    "type": "object",
                    "properties": {"data_path": {"type": "string"}},
                },
            },
        }]
        await client.chat(messages=[{"role": "user", "content": "QC my data"}], tools=openai_tools)

        converted = delivered_kwargs["tools"]
        assert len(converted) == 1
        assert converted[0]["name"] == "qc_report"
        assert "input_schema" in converted[0]
        assert "function" not in converted[0]  # OpenAI wrapper stripped

    @pytest.mark.asyncio
    async def test_empty_messages_list(self):
        """chat() handles empty messages list gracefully."""
        client = _make_anthropic_client()

        async def _fake_create(**kw: Any) -> Any:
            return _MockAnthropicResponse()

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        result = await client.chat(messages=[])
        assert result["content"] == [{"type": "text", "text": "Hello."}]

    @pytest.mark.asyncio
    async def test_config_defaults_used_when_not_overridden(self):
        """Config max_tokens/temperature used when call doesn't specify."""
        client = _make_anthropic_client(max_tokens=4096, temperature=0.3)
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockAnthropicResponse()

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert delivered_kwargs["max_tokens"] == 4096
        assert delivered_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_tool_only_response(self):
        """chat() with only tool_use blocks (no text)."""
        client = _make_anthropic_client()
        mock_resp = _MockAnthropicResponse(
            content=[
                _MockAnthropicToolUseBlock(id="t1", name="run_r", input={"code": "1+1"}),
            ],
            stop_reason="tool_use",
        )

        async def _fake_create(**kw: Any) -> Any:
            return mock_resp

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        result = await client.chat(messages=[{"role": "user", "content": "Run 1+1"}])
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "run_r"

    @pytest.mark.asyncio
    async def test_api_error_propagates(self):
        """Anthropic API errors are not swallowed."""
        client = _make_anthropic_client()

        async def _fake_create(**kw: Any) -> Any:
            msg = "Invalid API key"
            raise ValueError(msg)

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="Invalid API key"):
            await client.chat(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_max_tokens_zero_not_replaced_by_default(self):
        """max_tokens=0 is not falsy — it should be passed through, not replaced by config default."""
        client = _make_anthropic_client(max_tokens=16384)
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockAnthropicResponse()

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=0,
        )
        assert delivered_kwargs["max_tokens"] == 0

    @pytest.mark.asyncio
    async def test_thinking_budget_injected_when_configured(self):
        """When thinking_budget_tokens > 0, thinking block and adjusted max_tokens are sent."""
        client = _make_anthropic_client(thinking_budget_tokens=4096, max_tokens=16384)
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockAnthropicResponse()

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(messages=[{"role": "user", "content": "Think deeply."}])
        assert delivered_kwargs.get("thinking") == {
            "type": "enabled",
            "budget_tokens": 4096,
        }
        # max_tokens should be at least thinking_budget_tokens + 1024
        assert delivered_kwargs["max_tokens"] >= 4096 + 1024

    @pytest.mark.asyncio
    async def test_thinking_not_injected_when_budget_is_zero(self):
        """When thinking_budget_tokens=0, no thinking block is sent."""
        client = _make_anthropic_client(thinking_budget_tokens=0)
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockAnthropicResponse()

        client._client.messages.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert "thinking" not in delivered_kwargs


# ═══════════════════════════════════════════════════════════════════
# OpenAI mock helpers
# ═══════════════════════════════════════════════════════════════════


class _MockOpenAIFunctionCall:
    def __init__(self, id: str = "call_1", name: str = "read_file", arguments: str = '{"path": "/f.csv"}') -> None:
        self.id = id
        self.function = _MockOpenAIFunc(name, arguments)


class _MockOpenAIFunc:
    def __init__(self, name: str = "read_file", arguments: str = '{"path": "/f.csv"}') -> None:
        self.name = name
        self.arguments = arguments


class _MockOpenAIChoice:
    def __init__(
        self,
        content: str | None = "Response text.",
        tool_calls: list[_MockOpenAIFunctionCall] | None = None,
        finish_reason: str = "stop",
    ) -> None:
        self.message = _MockOpenAIMessage(content, tool_calls)
        self.finish_reason = finish_reason


class _MockOpenAIMessage:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[_MockOpenAIFunctionCall] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _MockOpenAIUsage:
    def __init__(self, prompt_tokens: int = 200, completion_tokens: int = 80) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _MockOpenAIResponse:
    def __init__(
        self,
        choices: list[_MockOpenAIChoice] | None = None,
        usage: _MockOpenAIUsage | None = None,
        model: str = "gpt-4o",
    ) -> None:
        self.choices = choices or [_MockOpenAIChoice()]
        self.usage = usage or _MockOpenAIUsage()
        self.model = model


def _make_openai_client(**kwargs: Any) -> OpenAIClient:
    """Create an OpenAIClient with a mocked SDK, skipping real API init."""
    cfg = OpenAIConfig(api_key=kwargs.pop("api_key", "sk-openai-test"), **kwargs)
    client = OpenAIClient.__new__(OpenAIClient)
    LLMClient.__init__(client, cfg)
    client._model = cfg.model
    client._client = MagicMock()
    return client


# ═══════════════════════════════════════════════════════════════════
# OpenAIClient.chat() tests
# ═══════════════════════════════════════════════════════════════════


class TestOpenAIClientChat:
    """OpenAIClient.chat() — mocked API responses."""

    @pytest.mark.asyncio
    async def test_text_only_response(self):
        """chat() returns text content from a standard completion."""
        client = _make_openai_client()
        mock_resp = _MockOpenAIResponse(
            choices=[_MockOpenAIChoice(content="Your Q3 revenue grew 12%.")],
        )

        async def _fake_create(**kw: Any) -> Any:
            return mock_resp

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        result = await client.chat(messages=[{"role": "user", "content": "Analyze Q3"}])
        assert result["content"] == [{"type": "text", "text": "Your Q3 revenue grew 12%."}]
        assert result["tool_calls"] is None
        assert result["finish_reason"] == "stop"
        assert result["usage"]["input_tokens"] == 200
        assert result["usage"]["output_tokens"] == 80

    @pytest.mark.asyncio
    async def test_response_with_tool_calls_json_args(self):
        """Tool call arguments returned as JSON strings are parsed to dict."""
        client = _make_openai_client()
        mock_resp = _MockOpenAIResponse(
            choices=[
                _MockOpenAIChoice(
                    content=None,
                    tool_calls=[
                        _MockOpenAIFunctionCall(
                            id="call_abc",
                            name="differential_analysis",
                            arguments='{"count_matrix": "/counts.csv", "method": "deseq2"}',
                        ),
                    ],
                    finish_reason="tool_calls",
                ),
            ],
        )

        async def _fake_create(**kw: Any) -> Any:
            return mock_resp

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        result = await client.chat(messages=[{"role": "user", "content": "Run DE analysis"}])
        assert result["tool_calls"] == [
            {
                "id": "call_abc",
                "name": "differential_analysis",
                "input": {"count_matrix": "/counts.csv", "method": "deseq2"},
            },
        ]
        assert result["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_tool_calls_malformed_json_preserved_as_string(self):
        """Tool call args that aren't valid JSON are kept as raw string."""
        client = _make_openai_client()
        mock_resp = _MockOpenAIResponse(
            choices=[
                _MockOpenAIChoice(
                    content=None,
                    tool_calls=[
                        _MockOpenAIFunctionCall(
                            id="call_bad",
                            name="run_python",
                            arguments="not valid json {{{",
                        ),
                    ],
                    finish_reason="tool_calls",
                ),
            ],
        )

        async def _fake_create(**kw: Any) -> Any:
            return mock_resp

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        result = await client.chat(messages=[{"role": "user", "content": "Run code"}])
        assert result["tool_calls"][0]["input"] == "not valid json {{{"

    @pytest.mark.asyncio
    async def test_system_prompt_prepended_as_message(self):
        """System prompt is injected as first message with role='system'."""
        client = _make_openai_client()
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockOpenAIResponse()

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            system="System instructions.",
        )
        msgs = delivered_kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "System instructions."}
        assert msgs[1] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_custom_temperature_and_max_tokens(self):
        """Overrides are forwarded to the OpenAI API call."""
        client = _make_openai_client()
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockOpenAIResponse()

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=1.2,
            max_tokens=8000,
        )
        assert delivered_kwargs["temperature"] == 1.2
        assert delivered_kwargs["max_tokens"] == 8000

    @pytest.mark.asyncio
    async def test_tool_choice_auto_when_tools_provided(self):
        """tool_choice='auto' is set when tools are passed."""
        client = _make_openai_client()
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockOpenAIResponse()

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(
            messages=[{"role": "user", "content": "Run tool"}],
            tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        )
        assert delivered_kwargs["tools"] is not None
        assert delivered_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_no_tool_choice_when_no_tools(self):
        """tool_choice is NOT set when no tools are provided."""
        client = _make_openai_client()
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockOpenAIResponse()

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert "tool_choice" not in delivered_kwargs

    @pytest.mark.asyncio
    async def test_missing_usage_handled_gracefully(self):
        """When API returns None usage, defaults to 0."""
        client = _make_openai_client()
        mock_resp = _MockOpenAIResponse(usage=None)  # type: ignore[arg-type]
        mock_resp.usage = None

        async def _fake_create(**kw: Any) -> Any:
            return mock_resp

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        result = await client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0

    @pytest.mark.asyncio
    async def test_config_defaults_used_when_not_overridden(self):
        """Config values are used when chat() doesn't specify overrides."""
        client = _make_openai_client(max_tokens=2048, temperature=0.5)
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockOpenAIResponse()

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert delivered_kwargs["max_tokens"] == 2048
        assert delivered_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_content_none_with_tool_calls(self):
        """When message.content is None but tool_calls exist, content is empty."""
        client = _make_openai_client()
        mock_resp = _MockOpenAIResponse(
            choices=[
                _MockOpenAIChoice(
                    content=None,
                    tool_calls=[
                        _MockOpenAIFunctionCall(id="c1", name="gsea_analysis", arguments='{"gene_list": "brca1,tp53"}'),
                    ],
                    finish_reason="tool_calls",
                ),
            ],
        )

        async def _fake_create(**kw: Any) -> Any:
            return mock_resp

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        result = await client.chat(messages=[{"role": "user", "content": "Run GSEA"}])
        # No text block when content is None
        texts = [b["text"] for b in result["content"] if b["type"] == "text"]
        assert len(texts) == 0
        assert len(result["tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_api_error_propagates(self):
        """OpenAI API errors are not swallowed."""
        client = _make_openai_client()

        async def _fake_create(**kw: Any) -> Any:
            msg = "Rate limit exceeded"
            raise RuntimeError(msg)

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            await client.chat(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_max_tokens_zero_not_replaced_by_default(self):
        """max_tokens=0 is not falsy — it should be passed through, not fall back to config default."""
        client = _make_openai_client(max_tokens=4096)
        delivered_kwargs: dict = {}

        async def _fake_create(**kw: Any) -> Any:
            nonlocal delivered_kwargs
            delivered_kwargs = kw
            return _MockOpenAIResponse()

        client._client.chat.completions.create = _fake_create  # type: ignore[attr-defined]

        await client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=0,
        )
        assert delivered_kwargs["max_tokens"] == 0
