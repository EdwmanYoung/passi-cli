# PassiAgent

AI-powered multi-omics bioinformatics analysis agent. Interactive LLM-driven analysis for genomics, transcriptomics, epigenetics, proteomics, metabolomics, and clinical statistics — with structured plan mode, task tracking, and full audit trail.

## Installation

```bash
git clone git@github.com:EdwmanYoung/passi-cli.git
cd passi-cli
pip install -e ".[dev]"
```

Python 3.10+ required. R (4.x) is auto-detected for Bioconductor methods; set `PASSI_EXECUTION__R_HOME` if auto-detection fails.

## Quick Start

```bash
# Copy and configure your API keys
cp .env.example .env
# Edit .env with your DeepSeek API key

# Interactive chat (Rich TUI)
passi chat

# Single query (stdout)
passi ask "Run differential expression analysis on my counts.csv"

# List available tools
passi tool list

# List analysis methods
passi knowledge methods --domain transcriptomics

# Manage sessions
passi session list
```

### Provider Configuration

PassiAgent supports three LLM providers via `PASSI_*` environment variables or YAML/JSON config files:

| Provider | Env Prefix | Endpoint |
|----------|-----------|----------|
| `anthropic` | `PASSI_ANTHROPIC__*` | Anthropic Messages API (or DeepSeek compatible) |
| `openai` | `PASSI_OPENAI__*` | OpenAI Chat Completions API (or DeepSeek compatible) |
| `ollama` | `PASSI_OLLAMA__*` | Local Ollama server (OpenAI-compatible) |

Provider-specific config fields: `API_KEY`, `BASE_URL`, `MODEL`, `MAX_TOKENS`, `TOOL_CALL_MAX_TOKENS`, `TEMPERATURE`. The default provider is set via `PASSI_DEFAULT_PROVIDER`.

### R Environment

R is used for Bioconductor packages (DESeq2, edgeR, limma, WGCNA, mixOmics, MOFA2, survival, etc.). Auto-detection checks: explicit config → `PASSI_R_HOME` / `R_HOME` env vars → project-local `./R/` directory → system `Program Files/R`. Falls back to `Rscript` subprocess if rpy2 is unavailable.

## Architecture

```
User Input → PassiAgent (ReAct Loop)
               ├── LLMClient (Anthropic/OpenAI/Ollama) → Reasoning + Tool Selection
               ├── ToolRegistry → Execute Tools (Python/R subprocess)
               ├── PlanManager → Structured Analysis Plan (plan.yaml)
               ├── TaskTracker → Execution Records (tasks.jsonl)
               ├── ProvenanceTracker → Reproducibility (provenance.jsonl)
               └── Wire → Event Bus + Audit Log (wire.jsonl)
```

**ReAct loop** (max 20 iterations): LLM reasons → selects tools → tools execute → results fed back → repeat until done. Each iteration uses `tool_call_max_tokens` (4096 default) for tool selection; final responses use `max_tokens` (16384 default).

**Audit trail** (5 layers per session under `.passi/sessions/{id}/`):

| File | Content |
|------|---------|
| `session.yaml` | Session metadata, domain, message count |
| `plan.yaml` | Structured analysis plan with step statuses |
| `wire.jsonl` | All agent↔tool↔user communication events |
| `tasks.jsonl` | Per-tool execution records with timing |
| `provenance.jsonl` | Tool invocations with file checksums |

A global/fallback `wire.jsonl` is also written to `.passi/wire.jsonl`.

### Tool Categories

| Category | Tools |
|----------|-------|
| **system** | `create_plan`, `update_plan_status`, `get_plan` |
| **io** | `read_file`, `write_file`, `parse_omics_data`, `export_results` |
| **exec** | `run_python`, `run_r` |
| **qc** | `qc_report` |
| **genomics** | `vcf_stats`, `gwas_analysis`, `manhattan_plot` |
| **epigenetics** | `peak_qc`, `methylation_analysis` |
| **transcriptomics** | `differential_analysis`, `enrichment` |
| **clinical** | `survival_analysis` |

Tools extend `CallableTool[ParamsT]` — define a Pydantic params model, `name`, `description`, and `async execute()`. OpenAI/Anthropic function-calling schemas are auto-generated from the params model.

### Project Layout

```
digitagent/
├── src/passi/          # Source code
├── docs/               # Architecture & design docs
├── scripts/            # Setup helpers
├── .passi/             # Project-local config + runtime data (gitignored)
│   ├── settings.yaml   # Project-level settings
│   ├── hooks.yaml      # User event hooks
│   ├── sessions/       # Session directories
│   ├── e2e_results/    # End-to-end test outputs
│   └── wire.jsonl      # Global audit log
├── result/             # Analysis outputs (configurable via result_dir)
├── data/               # User data (configurable via data_dir, gitignored)
└── test_dataset/       # Public omics test data
```

`.passi/`, `result/`, and `data/` are gitignored by default because they contain generated outputs, runtime data, and potentially large user data. Keep source code and docs in `src/` and `docs/`. If you want to share project-level defaults, explicitly add `.passi/settings.yaml` and/or `.passi/hooks.yaml` to version control.

### Plan Mode

For complex multi-step analyses, the LLM first creates a structured plan (`create_plan`), then executes step-by-step with progress tracking (`update_plan_status`). Users can review and approve plans before execution.

### Knowledge Base

Built-in catalog of 50+ bioinformatics methods and 40+ file format definitions, queryable via `passi knowledge` commands.

## Development

```bash
# Run tests
python -m pytest src/tests/ -q -m "not integration"

# Run integration tests (requires valid API key in .env)
python -m pytest src/tests/test_infra/test_llm_integration.py -v -m integration

# Run integration tests excluding slow ones
python -m pytest src/tests/test_infra/test_llm_integration.py -v -m "integration and not slow"

# Lint & type check
ruff check src/
mypy src/
```

Tests use `FakeLLMClient` for agent tests and `MagicMock`-based SDK mocks for LLM client tests. Integration tests are marked `@pytest.mark.integration` and auto-skip when no valid API key is configured.

## Data

A companion test dataset collection is available in `test_dataset/` covering 8 omics/statistics domains with real public data from GEO, PRIDE, ENCODE, TCGA, MetaboLights, NHANES, and more. See `test_dataset/README.md` for details.

## License

MIT
