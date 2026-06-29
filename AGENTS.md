# PassiAgent — Agent Guide

> This file is written for AI coding agents. It summarizes the project structure, conventions, build/test commands, and security considerations based on the actual codebase. When in doubt, prefer the source code in `src/passi/` over the higher-level design documents in `docs/`.

## Project Overview

PassiAgent (`passi`) is a Python CLI application for multi-omics bioinformatics downstream analysis. It exposes an interactive, LLM-driven ReAct agent that can:

- Parse and preview common omics file formats (VCF, BED, FASTA, count matrices, h5ad, etc.).
- Run differential expression (DESeq2/edgeR/limma), gene set enrichment (fgsea/clusterProfiler), survival analysis (Kaplan–Meier / Cox PH), GWAS, peak QC, and methylation analysis.
- Execute arbitrary Python or R code in sandboxed subprocesses, with run directories preserved for audit.
- Maintain persistent analysis sessions with plans, tasks, provenance, and a wire event log.

The project name in `pyproject.toml` is `passi`, version `0.1.0`.

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ (code targets 3.10; mypy configured for 3.11) |
| Packaging | setuptools, `pyproject.toml` |
| CLI framework | Click |
| Interactive TUI | Rich + prompt-toolkit + Textual |
| LLM clients | Anthropic SDK, OpenAI SDK, Ollama-compatible endpoint |
| Data | pandas, numpy, pyarrow, h5py |
| Bioinformatics Python | scanpy, anndata, gseapy |
| Statistics/ML | scipy, statsmodels, scikit-learn |
| Visualization | matplotlib, seaborn |
| R bridge | rpy2 (optional) or `Rscript` subprocess fallback |
| Web API (stub) | FastAPI + uvicorn |
| Configuration | pydantic-settings, pyyaml, python-dotenv |
| Quality | ruff, mypy, pytest, pytest-asyncio, pytest-cov |

## Directory Layout (actual)

```
digitagent/
├── pyproject.toml              # Package metadata, deps, tool configs
├── README.md                   # Human-oriented overview
├── CLAUDE.md                   # Claude Code specific commands/conventions
├── AGENTS.md                   # This file
├── .env.example                # Template for LLM/R configuration
├── docs/
│   ├── architecture.md         # Layered architecture spec
│   ├── design.md               # Design principles & conventions (Chinese)
│   └── specification.md        # Software requirements spec (Chinese)
├── environments/               # Empty — reserved for conda env specs
├── pipelines/                  # Empty — reserved for user YAML workflows
├── scripts/
│   └── setup_r_portable.ps1    # PowerShell helper to install R-Portable
├── sessions/                   # Runtime session directories
├── result/                     # Analysis outputs + provenance.jsonl
├── output/                     # Legacy/generated analysis outputs
├── test_dataset/               # Multi-omics public test data collection
├── src/
│   ├── passi/                  # Main package
│   │   ├── main.py             # Click CLI entry point
│   │   ├── config.py           # PassiConfig & layered loading
│   │   ├── api/server.py       # FastAPI stub (health + root only)
│   │   ├── executors/
│   │   │   ├── __init__.py
│   │   │   └── r_executor.py   # rpy2 init + RExecutor
│   │   ├── infra/
│   │   │   ├── context.py      # Conversation context + compaction
│   │   │   ├── hooks.py        # User-configurable event hooks
│   │   │   ├── llm_client.py   # Anthropic/OpenAI/Ollama clients
│   │   │   ├── plan.py         # AnalysisPlan + PlanManager
│   │   │   ├── provenance.py   # ProvenanceTracker with checksums
│   │   │   ├── runtime.py      # DI container
│   │   │   ├── session.py      # SessionManager
│   │   │   └── task_tracker.py # Task execution records
│   │   ├── knowledge/
│   │   │   ├── formats.py      # File format registry + detection
│   │   │   ├── methods.py      # Analysis method catalog
│   │   │   └── pipelines.py    # Predefined pipeline definitions
│   │   ├── prompts/
│   │   │   ├── manager.py      # Template composition + skills
│   │   │   └── *.txt           # System prompt templates
│   │   ├── soul/
│   │   │   ├── protocol.py     # Soul ABC + AgentMessage
│   │   │   └── passi_agent.py  # ReAct agent implementation
│   │   ├── tools/
│   │   │   ├── base.py         # CallableTool[ParamsT]
│   │   │   ├── registry.py     # ToolRegistry
│   │   │   ├── ask_user_tool.py
│   │   │   ├── clinical_tools.py
│   │   │   ├── enrichment_tools.py
│   │   │   ├── epigenetics_tools.py
│   │   │   ├── exec_tools.py   # run_python / run_r
│   │   │   ├── genomics_tools.py
│   │   │   ├── io_tools.py     # read_file / write_file / parse_omics_data / export_results
│   │   │   ├── qc_tools.py
│   │   │   ├── system_tools.py # create_plan / update_plan_status / get_plan
│   │   │   └── transcriptomics_tools.py
│   │   ├── ui/
│   │   │   ├── cli.py          # Rich TUI
│   │   │   ├── print_mode.py   # Non-interactive single-query mode
│   │   │   └── prompts.py      # UI prompt text
│   │   └── wire/
│   │       ├── protocol.py     # Wire event bus
│   │       └── persistence.py  # wire.jsonl read/export
│   └── tests/                  # Mirrors passi/ structure
│       ├── conftest.py
│       ├── fixtures/
│       │   ├── data_factories.py
│       │   └── mock_llm.py
│       ├── test_executors/
│       ├── test_infra/
│       ├── test_integration/
│       ├── test_knowledge/
│       ├── test_prompts/
│       ├── test_soul/
│       ├── test_tools/
│       ├── test_ui/
│       └── test_wire/
```

**Note:** Some items described in `docs/architecture.md` (sub-agents, `python_executor.py`, Docker sandbox, dedicated API routes/schemas) are **not yet implemented**. The actual tool execution is handled by `exec_tools.py` using `subprocess.Popen` and by `r_executor.py`.

## Architecture at a Glance

```
User ↔ UI (cli.py / print_mode.py / server stub)
        ↓
    Soul protocol (soul/protocol.py)
        ↓
    PassiAgent (soul/passi_agent.py) — ReAct loop (max 20 iterations)
        ├── LLMClient (infra/llm_client.py)
        ├── ToolRegistry (tools/registry.py)
        ├── ContextManager (infra/context.py)
        ├── PlanManager (infra/plan.py)
        ├── TaskTracker (infra/task_tracker.py)
        ├── ProvenanceTracker (infra/provenance.py)
        └── Wire event bus (wire/protocol.py)
```

All UI implementations depend on the `Soul` abstract class, not on `PassiAgent` directly.

## Configuration

Configuration is layered in `config.py` via `PassiConfig` (pydantic-settings, env prefix `PASSI_`, nested delimiter `__`).

Priority (lowest → highest):
1. Code defaults
2. `~/.passi/settings.yaml`
3. `<project>/.passi/settings.yaml`
4. CWD `.env`
5. CLI `--config` YAML/JSON
6. `PASSI_*` environment variables
7. `PASSI_CONFIG` JSON override

Key settings:
- `PASSI_DEFAULT_PROVIDER` — `anthropic`, `openai`, or `ollama`.
- `PASSI_ANTHROPIC__API_KEY`, `PASSI_ANTHROPIC__BASE_URL`, `PASSI_ANTHROPIC__MODEL`, etc.
- `PASSI_EXECUTION__R_HOME`, `PASSI_EXECUTION__R_LIB_PATH`, `PASSI_EXECUTION__RPY2_ENABLED`.
- `PASSI_SESSION__SESSIONS_DIR`.

R auto-detection order (when `r_home` is empty):
1. Explicit config / `PASSI_EXECUTION__R_HOME`
2. `PASSI_R_HOME` / `R_HOME` env vars
3. Project-local `./R/`
4. System R under `Program Files/R` (newest first)

## Build & Install Commands

```bash
# Editable install with dev dependencies
pip install -e ".[dev]"

# Install optional heavy bioinformatics deps
pip install -e ".[all]"
```

## Running Tests

```bash
# All unit tests (excludes integration tests that need real API keys)
python -m pytest src/tests/ -q -m "not integration"

# Single test file
python -m pytest src/tests/test_infra/test_config.py -v

# Single test
python -m pytest src/tests/test_infra/test_config.py::TestPassiConfigDefaults::test_default_provider -v

# Integration tests — require valid API keys in .env
python -m pytest src/tests/test_infra/test_llm_integration.py -v -m integration

# Integration tests excluding slow ones
python -m pytest src/tests/test_infra/test_llm_integration.py -v -m "integration and not slow"

# Coverage
pytest --cov=src/passi --cov-report=html --cov-report=term
```

Current status: 704 unit tests pass, 0 skipped, 18 deselected when excluding integration tests; project-local R environment is auto-configured for tests in `src/tests/conftest.py`.

## Lint & Type Check

```bash
ruff check src/passi
mypy src/
```

`pyproject.toml` configures:
- ruff target Python 3.10, line length 100, selects `E`, `F`, `I`, `N`, `W`, `UP`.
- mypy Python 3.11.

As of the latest check, `ruff check src/passi` reports many style issues (long lines, unused imports, etc.). Do not assume the codebase is lint-clean.

## CLI Usage

```bash
passi chat                    # Interactive Rich TUI
passi ask "query"             # Single query, stdout
passi afk "query"             # Autonomous mode (no user prompts)
passi tool list               # List registered tools
passi tool run <name> '{}'    # Execute a tool directly with JSON params
passi session list            # List sessions
passi session load <id>       # Restore a session
passi session delete <id>     # Delete a session
passi knowledge methods --domain transcriptomics
passi knowledge formats --domain genomics
passi server                  # FastAPI stub (reserved)
```

## Code Style Guidelines

- Python 3.10+ with modern annotations (`list[dict]`, `str | None`).
- Use `from __future__ import annotations` at the top of every module.
- All public functions and methods should have type annotations.
- Line length limit is 100 (ruff).
- Naming:
  - Classes: `PascalCase`
  - Functions/methods/modules: `snake_case`
  - Constants: `UPPER_SNAKE`
  - Private members: `_prefix`
- Pydantic models are preferred for params, config, and messages.
- Async-first: I/O operations use `async/await`.
- No global state: dependencies flow through `Runtime`.
- Tools return `{"success": bool, ...}` instead of raising exceptions for expected failures.
- Keep docstrings in English to match existing code.

## Tool System

Every tool extends `CallableTool[ParamsT]` (`tools/base.py`) and defines:

```python
class MyTool(CallableTool[MyParams]):
    name = "my_tool"
    description = "What it does and when to use it."
    params_model = MyParams

    async def execute(self, params: MyParams, **kwargs: Any) -> dict[str, Any]:
        return {"success": True, "result": ...}
```

To register a new tool:
1. Add it to the appropriate file under `src/passi/tools/` (or create a new file).
2. Import and register it in `PassiAgent._create_tool_registry()` in `src/passi/soul/passi_agent.py`.
3. If it is an analysis method, add an entry to `src/passi/knowledge/methods.py`.
4. Add tests under `src/tests/test_tools/` mirroring the production file name.

## Testing Strategy

- **TDD is expected**, especially for bug fixes: write a failing test, fix the code, verify the full suite.
- Unit tests use `FakeLLMClient` / `FakeLLMClientWithToolSequence` from `tests/fixtures/mock_llm.py`.
- Inject fakes via `agent._llm_client = fake_client` after construction.
- Integration tests are marked `@pytest.mark.integration` and auto-skip without valid API keys.
- Slow tests are marked `@pytest.mark.slow`.
- `pythonpath = ["src"]` in `pyproject.toml` lets tests import as `from passi...`.
- Test naming convention: `test_<method>_<condition>_<expected>`.

## R Environment

- Many analysis tools generate R scripts and execute them via `RExecutor`.
- Default mode uses `Rscript` subprocess; enable rpy2 with `PASSI_EXECUTION__RPY2_ENABLED=true`.
- On Windows, use `scripts/setup_r_portable.ps1` to install R-Portable into the project.
- Required Bioconductor packages include: `DESeq2`, `edgeR`, `limma`, `clusterProfiler`, `fgsea`, `WGCNA`, `mixOmics`, `MOFA2`, `survival`, `DSS`, `DiffBind`, `SNFtool`.

## Security Considerations

- **Code execution:** `run_python` and `run_r` execute arbitrary user-supplied code in subprocesses. They run as the current OS user and are not sandboxed. The Docker sandbox mentioned in architecture docs is not implemented.
- **API keys:** Store keys in `.env` or env vars; never hardcode them. `.env` is gitignored.
- **File access:** Tools read/write paths supplied by the LLM/user, restricted only by OS permissions. Avoid running against untrusted inputs without additional sandboxing.
- **Input length:** The codebase does not enforce a hard 100K character message limit automatically; callers / UI should enforce reasonable bounds.
- **Hooks:** User-configured hooks (`~/.passi/hooks.yaml`) can run shell commands or Python snippets on agent events; validate hook configs before enabling in sensitive environments.

## Known Limitations / Stubs

- `environments/` and `pipelines/` are empty directories.
- `src/passi/api/server.py` only exposes `/health` and `/`; routes, schemas, and WebSocket are not implemented.
- Sub-agents (`OmicsExpert`, `StatsExpert`, etc.) and `soul/subagents/` do not exist.
- `executors/python_executor.py` and `executors/sandbox.py` do not exist; Python execution lives in `tools/exec_tools.py`.
- Predefined pipelines in `knowledge/pipelines.py` are static definitions only; the `passi run` command loads YAML but does not execute pipeline steps end-to-end.

## Useful References

- `CLAUDE.md` — quick command cheat-sheet and testing conventions.
- `docs/architecture.md` — high-level component diagram and design rationale.
- `docs/design.md` — detailed design principles and TDD workflow (Chinese).
- `docs/specification.md` — full functional requirements and acceptance criteria (Chinese).
