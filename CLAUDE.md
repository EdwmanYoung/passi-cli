# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (dev)
pip install -e ".[dev]"

# Run all unit tests
python -m pytest src/tests/ -q -m "not integration"

# Run a single test file
python -m pytest src/tests/test_infra/test_config.py -v

# Run a single test
python -m pytest src/tests/test_infra/test_config.py::TestPassiConfigDefaults::test_default_provider -v

# Run integration tests (requires valid .env API keys)
python -m pytest src/tests/test_infra/test_llm_integration.py -v -m integration

# Run integration tests excluding slow ones
python -m pytest src/tests/test_infra/test_llm_integration.py -v -m "integration and not slow"

# Lint & type check
ruff check src/
mypy src/

# CLI
passi chat                    # Interactive TUI chat
passi ask "query"             # Single query, stdout
passi tool list               # List available tools
passi tool run <name> '{}'    # Execute a tool directly
passi session list            # List saved sessions
```

## Architecture

### Agent Layer (Soul)

`PassiAgent` implements the `Soul` ABC. Its `chat()` method runs a **ReAct loop** (max 20 iterations):

1. Build context from `ContextManager` (system prompt + tools + message history)
2. Call `LLMClient.chat()` — if no `tool_calls` returned, loop exits
3. For each tool call: create a `Task`, emit `Wire(TOOL_CALL)`, execute via `ToolRegistry`, record `Provenance`, complete task, emit `Wire(TOOL_RESULT)`, emit plan events
4. Check `needs_compaction()`, compact if needed

`chat_stream()` delegates to `chat()` internally and replays the result as individual stream events.

### Tool System

Every tool extends `CallableTool[ParamsT]` (in `tools/base.py`). Required attributes:

- `name: str` — unique, used by LLM and registry
- `description: str` — for LLM function-calling schemas
- `params_model: type[ParamsT]` — Pydantic `BaseModel` subclass for input validation
- `async execute(params: ParamsT, **kwargs) -> dict` — always returns `{"success": bool, ...}`

Schema export (`to_openai_schema()`, `to_anthropic_schema()`) is automatic from the params model — no manual schema writing. Tools are registered in `_create_tool_registry()` by category (io, exec, qc, genomics, epigenetics, transcriptomics, clinical, system).

### LLM Client Abstraction

`LLMClient(ABC)` defines `chat(messages, tools, system, temperature, max_tokens) -> dict`. All implementations return the same dict shape: `{"content": list[dict], "tool_calls": list|None, "usage": {"input_tokens", "output_tokens"}, "finish_reason": str, "model": str}`.

- **AnthropicClient** — system is a top-level param, tools converted from OpenAI to Anthropic format; supports extended thinking if `thinking_budget_tokens > 0`
- **OpenAIClient** — system injected as first message with `role: "system"`, `tool_choice: "auto"`
- **OllamaClient** — extends `OpenAIClient` (Ollama has OpenAI-compatible endpoint)

`create_llm_client(config, provider)` is the factory. All clients use async SDKs (`AsyncAnthropic`, `AsyncOpenAI`).

### Audit Trail (5 layers)

Execution produces this audit chain per session:

- **Wire** (`wire.jsonl`) — pub/sub event bus, synchronous emit to all listeners, persisted
- **Provenance** (`provenance.jsonl`) — per-tool-step records with checksums, exit codes, timing
- **Tasks** (`tasks.jsonl`) — per-tool-execution records, cross-linked to plan steps and provenance
- **Plan** (`plan.yaml`) — structured analysis plan with step statuses
- **Session** (`session.yaml`) — session metadata, checkpoints

Wire event types: `USER_MESSAGE`, `AGENT_MESSAGE`, `TOOL_CALL`, `TOOL_RESULT`, `SESSION_START/END`, `PLAN_CREATED`, `PLAN_STEP_START/COMPLETE/FAILED`, etc.

### Config (`PassiConfig`)

Uses `pydantic-settings` with `env_prefix="PASSI_"`, `env_nested_delimiter="__"`. Precedence: env vars > `PASSI_CONFIG` JSON > YAML/JSON file > `.env` file > defaults.

Nested config classes: `AnthropicConfig`, `OpenAIConfig`, `OllamaConfig` (all extend `LLMProviderConfig`), `ExecutionConfig`, `SessionConfig`. Provider selection via `config.get_llm_config(provider)`.

Key defaults: `AnthropicConfig.max_tokens=16384`, `tool_call_max_tokens=4096` (per ReAct iteration), `thinking_budget_tokens=0` (disabled).

### Runtime

`Runtime` is a DI container (`@dataclass`) that lazy-initializes shared services: `SessionManager`, `ContextManager`, `LLMClient` (cached by provider). The `PassiAgent` receives a `Runtime` and pulls services from it rather than creating them directly.

### Testing

**TDD required for bug fixes.** When fixing a bug:
1. Write a failing test that reproduces the bug
2. Verify the test fails with the current code
3. Apply the fix
4. Verify the test passes
5. Run the full suite

- Unit tests use `FakeLLMClient` / `FakeLLMClientWithToolSequence` from `tests/fixtures/mock_llm.py` — inject via `agent._llm_client = fake_client` after construction
- `_make_anthropic_client()` / `_make_openai_client()` in `test_llm_client.py` use `__new__` bypass + `MagicMock()` to skip real SDK init
- Integration tests (marked `@pytest.mark.integration`) require real API keys in `.env`; skip when `_is_api_configured()` returns False
- Slow tests are marked `@pytest.mark.slow` and run separately
- `pythonpath = ["src"]` in `pyproject.toml` so test imports use `from passi.` without editable install
