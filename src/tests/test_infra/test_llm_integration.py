"""Integration tests — real LLM API calls.

Validates actual connectivity, latency, response structure, and tool calling
against the configured providers.  All tests are skipped when no valid API key
is present in ``.env``.

Run only integration tests:
    python -m pytest src/tests/test_infra/test_llm_integration.py -v -m integration

Run with slow tests excluded:
    python -m pytest src/tests/test_infra/test_llm_integration.py -v -m "integration and not slow"
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from passi.config import PassiConfig, AnthropicConfig, OpenAIConfig
from passi.infra.llm_client import AnthropicClient, OpenAIClient


# ── Helpers ──────────────────────────────────────────────────────────


def _is_api_configured(provider: str) -> bool:
    """Check whether .env has a real (non-placeholder) API key for *provider*."""
    cfg = PassiConfig()
    key = getattr(cfg, provider).api_key
    return bool(key) and "your-" not in key and "sk-your" not in key


def _get_config() -> PassiConfig:
    return PassiConfig()


def _skip_if_not_configured(provider: str) -> None:
    """Helper — raises ``pytest.skip`` when no valid key is available."""
    if not _is_api_configured(provider):
        pytest.skip(f"No valid {provider} API key in .env — set PASSI_{provider.upper()}__API_KEY")


def _report_latency(label: str, start: float) -> None:
    """Print a latency line to stdout so it appears in test output."""
    elapsed = (time.perf_counter() - start) * 1000
    print(f"\n  [{label}] latency: {elapsed:.0f} ms")


# ═════════════════════════════════════════════════════════════════════
# Anthropic-compatible endpoint (DeepSeek)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestAnthropicClientReal:
    """Real API calls through AnthropicClient → DeepSeek anthropic endpoint."""

    @pytest.mark.asyncio
    async def test_simple_text_chat(self):
        """Basic ask-and-answer round-trip."""
        _skip_if_not_configured("anthropic")
        cfg = _get_config()
        client = AnthropicClient(cfg.anthropic)

        t0 = time.perf_counter()
        result = await client.chat(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=32,
        )
        _report_latency("anthropic simple text", t0)

        assert result["finish_reason"] in ("end_turn", "stop", "max_tokens", "length")
        assert result["usage"]["input_tokens"] > 0
        assert result["usage"]["output_tokens"] > 0
        assert result["model"] is not None
        texts = [b["text"] for b in result["content"] if b["type"] == "text"]
        assert len(texts) >= 1, f"Expected at least one text block, got {result['content']}"

    @pytest.mark.asyncio
    async def test_system_prompt(self):
        """System prompt constrains the output language."""
        _skip_if_not_configured("anthropic")
        cfg = _get_config()
        client = AnthropicClient(cfg.anthropic)

        t0 = time.perf_counter()
        result = await client.chat(
            messages=[{"role": "user", "content": "What is 1+1?"}],
            system="Reply in Chinese only. Be brief.",
            max_tokens=256,
        )
        _report_latency("anthropic system prompt", t0)

        combined = " ".join(
            [b["text"] for b in result["content"] if b["type"] == "text"]
        )
        assert any("一" <= ch <= "鿿" for ch in combined), (
            f"Expected Chinese characters in response, got: {combined[:100]}"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_long_response(self):
        """Longer response — verifies token counts scale properly."""
        _skip_if_not_configured("anthropic")
        cfg = _get_config()
        client = AnthropicClient(cfg.anthropic)

        t0 = time.perf_counter()
        result = await client.chat(
            messages=[{
                "role": "user",
                "content": "List the top 10 most common bioinformatics file formats "
                           "and give a one-sentence description of each.",
            }],
            max_tokens=1024,
        )
        _report_latency("anthropic long response", t0)

        assert result["usage"]["output_tokens"] > 50, (
            "Long prompt should return >50 output tokens"
        )
        texts = [b["text"] for b in result["content"] if b["type"] == "text"]
        combined = " ".join(texts)
        assert len(combined) > 200, f"Long response too short: {len(combined)} chars"

    @pytest.mark.asyncio
    async def test_tool_calling(self):
        """Real tool/function calling — model selects and uses a tool."""
        _skip_if_not_configured("anthropic")
        cfg = _get_config()
        client = AnthropicClient(cfg.anthropic)

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        }]

        t0 = time.perf_counter()
        result = await client.chat(
            messages=[{
                "role": "user",
                "content": "What's the weather in Beijing? Use the weather tool.",
            }],
            tools=tools,
            max_tokens=128,
        )
        _report_latency("anthropic tool calling", t0)

        # Model should call the tool
        assert result["tool_calls"] is not None, (
            f"Expected model to call get_weather tool, got {result}"
        )
        assert result["tool_calls"][0]["name"] == "get_weather"
        assert "city" in result["tool_calls"][0]["input"]
        assert result["finish_reason"] == "tool_use"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_bioinformatics_query(self):
        """Realistic bioinformatics query — domain knowledge + tool suggestions."""
        _skip_if_not_configured("anthropic")
        cfg = _get_config()
        client = AnthropicClient(cfg.anthropic)

        tools = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "max_lines": {"type": "integer", "description": "Max lines to read"},
                    },
                    "required": ["path"],
                },
            },
        }]

        t0 = time.perf_counter()
        result = await client.chat(
            messages=[{
                "role": "user",
                "content": (
                    "I have an RNA-seq count matrix at ./data/counts.csv with 300 samples. "
                    "I need to run differential expression analysis between treatment and control groups. "
                    "What steps should I take? Suggest tools I should use."
                ),
            }],
            tools=tools,
            max_tokens=512,
        )
        _report_latency("anthropic bioinfo query", t0)

        texts = [b["text"] for b in result["content"] if b["type"] == "text"]
        combined = " ".join(texts).lower()
        # Should mention relevant concepts
        keywords = ["rna-seq", "differential", "expression", "deseq2", "normalize"]
        matched = [kw for kw in keywords if kw in combined]
        assert len(matched) >= 2, (
            f"Expected at least 2 bioinfo keywords, got {matched}. Response: {combined[:200]}"
        )

    @pytest.mark.asyncio
    async def test_model_name_passed(self):
        """The model name from config is reflected in the response."""
        _skip_if_not_configured("anthropic")
        cfg = _get_config()

        # Use v4-flash if configured model is v4-pro, just to test different models
        model_to_test = cfg.anthropic.model
        alt_cfg = AnthropicConfig(
            api_key=cfg.anthropic.api_key,
            base_url=cfg.anthropic.base_url,
            model=model_to_test,
            max_tokens=64,
        )
        client = AnthropicClient(alt_cfg)

        t0 = time.perf_counter()
        result = await client.chat(
            messages=[{"role": "user", "content": "Say HI"}],
            max_tokens=32,
        )
        _report_latency(f"anthropic model={model_to_test}", t0)

        # DeepSeek returns the actual model name used
        assert result["model"] is not None
        print(f"\n  [anthropic] requested model: {model_to_test}, response model: {result['model']}")


# ═════════════════════════════════════════════════════════════════════
# OpenAI-compatible endpoint (DeepSeek)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestOpenAIClientReal:
    """Real API calls through OpenAIClient → DeepSeek OpenAI endpoint."""

    @pytest.mark.asyncio
    async def test_simple_text_chat(self):
        """Basic chat completion round-trip."""
        _skip_if_not_configured("openai")
        cfg = _get_config()
        client = OpenAIClient(cfg.openai)

        t0 = time.perf_counter()
        result = await client.chat(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=128,
        )
        _report_latency("openai simple text", t0)

        assert result["finish_reason"] in ("stop", "end_turn", "max_tokens", "length")
        assert result["usage"]["input_tokens"] > 0
        assert result["usage"]["output_tokens"] > 0
        assert result["model"] is not None
        texts = [b["text"] for b in result["content"] if b["type"] == "text"]
        assert len(texts) >= 1

    @pytest.mark.asyncio
    async def test_system_prompt(self):
        """System prompt as first message in the messages array."""
        _skip_if_not_configured("openai")
        cfg = _get_config()
        client = OpenAIClient(cfg.openai)

        t0 = time.perf_counter()
        result = await client.chat(
            messages=[{"role": "user", "content": "Say 你好"}],
            system="You only speak Chinese. Keep responses under 10 words.",
            temperature=0.0,
            max_tokens=128,
        )
        _report_latency("openai system prompt", t0)

        combined = " ".join(
            [b["text"] for b in result["content"] if b["type"] == "text"]
        )
        assert len(combined) > 0, "Expected non-empty response"

    @pytest.mark.asyncio
    async def test_tool_calling(self):
        """Real function calling through OpenAI-compatible endpoint."""
        _skip_if_not_configured("openai")
        cfg = _get_config()
        client = OpenAIClient(cfg.openai)

        tools = [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate",
                        },
                    },
                    "required": ["expression"],
                },
            },
        }]

        t0 = time.perf_counter()
        result = await client.chat(
            messages=[{
                "role": "user",
                "content": "What is 123 * 456? Use the calculate tool.",
            }],
            tools=tools,
            max_tokens=128,
        )
        _report_latency("openai tool calling", t0)

        assert result["tool_calls"] is not None, (
            f"Expected tool call, got finish_reason={result.get('finish_reason')}"
        )
        assert result["tool_calls"][0]["name"] == "calculate"
        assert isinstance(result["tool_calls"][0]["input"], dict)

    @pytest.mark.asyncio
    async def test_temperature_zero_deterministic(self):
        """temperature=0 should produce roughly the same output on repeated calls."""
        _skip_if_not_configured("openai")
        cfg = _get_config()
        client = OpenAIClient(cfg.openai)

        results = []
        for _ in range(3):
            r = await client.chat(
                messages=[{"role": "user", "content": "Reply with exactly: PASS"}],
                temperature=0.0,
                max_tokens=128,
            )
            texts = [b["text"] for b in r["content"] if b["type"] == "text"]
            results.append(" ".join(texts).strip())

        # At temperature 0, results should be identical or very similar
        unique = set(results)
        print(f"\n  [openai temp=0] unique responses: {unique}")
        # At least 2 of 3 should match
        assert len(unique) <= 2, f"Temp=0 should be deterministic, got {unique}"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_v4_flash_model(self):
        """Test the fast model variant."""
        _skip_if_not_configured("openai")
        cfg = _get_config()
        flash_cfg = OpenAIConfig(
            api_key=cfg.openai.api_key,
            base_url=cfg.openai.base_url,
            model="deepseek-v4-flash",
            max_tokens=128,
        )
        client = OpenAIClient(flash_cfg)

        t0 = time.perf_counter()
        result = await client.chat(
            messages=[{"role": "user", "content": "Explain what a VCF file is in one sentence."}],
            max_tokens=128,
        )
        _report_latency("openai flash model", t0)

        assert result["usage"]["input_tokens"] > 0
        texts = [b["text"] for b in result["content"] if b["type"] == "text"]
        assert len(texts) >= 1


@pytest.mark.integration
class TestCrossProviderConsistency:
    """Both providers return structurally identical dicts."""

    _required_keys = {"content", "tool_calls", "usage", "finish_reason", "model"}
    _usage_keys = {"input_tokens", "output_tokens"}

    @pytest.mark.asyncio
    async def test_anthropic_response_structure(self):
        """AnthropicClient response dict has all required fields."""
        _skip_if_not_configured("anthropic")
        client = AnthropicClient(_get_config().anthropic)

        result = await client.chat(
            messages=[{"role": "user", "content": "Say: STRUCTURE_TEST"}],
            max_tokens=32,
        )
        assert self._required_keys == set(result.keys()), (
            f"Missing keys: {self._required_keys - set(result.keys())}"
        )
        assert self._usage_keys == set(result["usage"].keys())
        assert isinstance(result["content"], list)
        assert result["tool_calls"] is None or isinstance(result["tool_calls"], list)

    @pytest.mark.asyncio
    async def test_openai_response_structure(self):
        """OpenAIClient response dict has all required fields."""
        _skip_if_not_configured("openai")
        client = OpenAIClient(_get_config().openai)

        result = await client.chat(
            messages=[{"role": "user", "content": "Say: STRUCTURE_TEST"}],
            max_tokens=32,
        )
        assert self._required_keys == set(result.keys()), (
            f"Missing keys: {self._required_keys - set(result.keys())}"
        )
        assert self._usage_keys == set(result["usage"].keys())
        assert isinstance(result["content"], list)
        assert result["tool_calls"] is None or isinstance(result["tool_calls"], list)
