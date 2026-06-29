# PassiAgent — System Architecture Specification

> **Version:** 0.1.0 | **Date:** 2026-06-25 | **Status:** Draft

## 1. Overview

PassiAgent is a Python-based passi agent for multi-omics bioinformatics downstream analysis. It assists researchers with single-omics and multi-omics integrated data analysis through a CLI-driven conversational interface, with reserved web API and client SDK interfaces.

### 1.1 Scope

The system covers six omics domains:
- **Genomics** — GWAS, variant analysis, CNV calling
- **Epigenetics** — ChIP-seq/ATAC-seq peak analysis, methylation (WGBS/RRBS)
- **Transcriptomics** — bulk RNA-seq (DESeq2/edgeR/limma), single-cell (Scanpy/Seurat)
- **Proteomics** — DDA/DIA quantification, differential abundance
- **Metabolomics** — LC-MS/GC-MS peak alignment, pathway mapping
- **Clinical Statistics** — survival analysis, regression, meta-analysis

### 1.2 Design References

Architecture inspired by Kimi CLI's layered pattern: Soul protocol, Wire communication, Runtime DI, tool-first design.

---

## 2. Layered Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       UI Layer                               │
│   Rich TUI (CLI)  │  Print Mode (stdout)  │  REST / WS API  │
├──────────────────────────────────────────────────────────────┤
│                   Orchestration Layer                        │
│   SessionManager  │  WorkflowEngine  │  WireProto (JSON-RPC) │
├──────────────────────────────────────────────────────────────┤
│                      Agent Layer                             │
│   Soul Protocol ── PassiAgent ── Sub-agents                │
│   ┌──────────┐  ┌───────────┐  ┌──────────────────┐         │
│   │  Omics   │  │   Stats   │  │  MultiOmics      │         │
│   │  Expert  │  │  Expert   │  │  Integrator      │         │
│   └──────────┘  └───────────┘  └──────────────────┘         │
├──────────────────────────────────────────────────────────────┤
│                  Tool & Execution Layer                      │
│   ToolRegistry │ PyExecutor │ RExecutor │ DockerSandbox      │
│   ┌────────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐  │
│   │ IO Tools   │ │ QC Tools │ │ Domain  │ │ Integration  │  │
│   │            │ │          │ │ Tools   │ │ Tools        │  │
│   ├────────────┤ ├──────────┤ ├─────────┤ ├──────────────┤  │
│   │ Viz Tools  │ │ Exec     │ │ Search  │ │ Pipeline     │  │
│   └────────────┘ └──────────┘ └─────────┘ └──────────────┘  │
├──────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                       │
│   Config │ Session │ Context │ LLM Client │ Provenance       │
│   ┌──────┐ ┌──────┐ ┌───────┐ ┌──────────┐ ┌─────────────┐ │
│   │ YAML │ │ Dir  │ │ Token │ │ Anthropic │ │ Wire.jsonl  │ │
│   │ ENV  │ │ Per  │ │ Win.  │ │ OpenAI    │ │ Checksums   │ │
│   │ .env │ │ Sess │ │ Mgr.  │ │ Ollama    │ │ Reports     │ │
│   └──────┘ └──────┘ └───────┘ └──────────┘ └─────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 Layer Responsibilities

| Layer | Responsibility | Key Components |
|-------|---------------|----------------|
| **UI** | User interaction surface | CLI TUI (Rich), Print mode, FastAPI server |
| **Orchestration** | Session lifecycle, workflow chaining, event routing | SessionManager, WorkflowEngine, Wire |
| **Agent** | LLM reasoning, tool selection, sub-agent delegation | PassiAgent, OmicsExpert, StatsExpert, MultiOmicsIntegrator |
| **Tool & Execution** | Validated tool execution in sandboxed environments | ToolRegistry, PyExecutor, RExecutor, DockerSandbox |
| **Infrastructure** | Config, state, context, LLM abstraction | Runtime (DI), ContextManager, LLMClient, ProvenanceTracker |

---

## 3. Core Components

### 3.1 Soul Protocol (`soul/protocol.py`)

Abstract interface that all UIs depend on. Defines three primitives:

```
Soul
├── chat(user_message) → AgentMessage        (complete response)
├── chat_stream(user_message) → Stream[Event] (streaming)
└── execute_tool(name, params) → AgentMessage (direct tool call)
```

**Principle:** UIs depend on `Soul`, never on `PassiAgent` directly. This enables swapping agent implementations without UI changes.

### 3.2 PassiAgent (`soul/passi_agent.py`)

The reference `Soul` implementation. Core agent loop (~200 lines):

```
1. Receive user message → add to context
2. Build full context (system prompt + messages + tool schemas)
3. Call LLM (via LLMClient abstraction)
4. If text response → emit to user, go to 7
5. If tool calls → execute via ToolRegistry → add results to context
6. Check context compaction → repeat from 2 (ReAct loop, max 20 iterations)
7. Return final AgentMessage
```

### 3.3 Tool System (`tools/`)

Each tool is a `CallableTool[ParamsT]` with:
- **Name + Description** — for LLM tool selection
- **Pydantic Params Model** — schema-validated input
- **async execute(params)** — implementation
- **schema export** — `to_openai_schema()` / `to_anthropic_schema()`

Tool execution pipeline:
```
Raw params → Pydantic validation → execute() → {success, result, [error]}
```

### 3.4 Wire Protocol (`wire/protocol.py`)

In-process pub/sub communication channel. All agent ↔ UI events flow through Wire.

Event types: `user_message`, `agent_message`, `agent_thinking`, `tool_call`, `tool_result`, `error`, `system`, `session_start`, `session_end`, `checkpoint`

Wire events are persisted to session-level `.passi/sessions/{id}/wire.jsonl` (or a global `.passi/wire.jsonl` fallback) for:
- **Session replay** — re-execute analysis from event log
- **Audit trail** — full provenance of every tool call
- **Debugging** — inspect agent reasoning chain

### 3.5 Runtime DI Container (`infra/runtime.py`)

Groups all shared services. Lazy initialization pattern:

```python
runtime = Runtime(config)
runtime.initialize()          # pre-warm all services
client = runtime.get_llm_client("anthropic")  # or "openai", "ollama"
session = runtime.session      # SessionManager (lazy)
context = runtime.context      # ContextManager (lazy)
```

### 3.6 LLM Client (`infra/llm_client.py`)

Multi-provider abstraction with unified `chat()` interface:
- **AnthropicClient** — Native tool use, 8K+ context
- **OpenAIClient** — Function calling, GPT-4o
- **OllamaClient** — Local models via OpenAI-compatible endpoint

### 3.7 Session Management (`infra/session.py`)

Each session is a directory under `.passi/sessions/` (the project-local Passi home directory):
```
.passi/
├── sessions/
│   └── session_20260624_143052/
│       ├── session.yaml          # SessionMeta (id, domain, timestamps)
│       ├── wire.jsonl            # Full communication log
│       ├── checkpoint_*.json     # State checkpoints
│       └── data/                 # Uploaded/session data
├── settings.yaml                 # Project-level settings
├── hooks.yaml                    # User event hooks
└── wire.jsonl                    # Global / fallback wire log
```

The Passi home directory is resolved as follows: if a `.passi/` directory exists in the current project, it is used; otherwise `~/.passi/` is used. This keeps per-project runtime data isolated from source code.

### 3.8 Context Manager (`infra/context.py`)

Token-aware context window with:
- Message tracking and token estimation
- Automatic compaction (summarize old messages when exceeding threshold)
- Tool schema management
- Checkpoint integration

---

## 4. Data Flow

### 4.1 Interactive Chat Flow

```
┌──────┐     ┌──────┐     ┌──────────────┐     ┌──────┐     ┌───────────┐
│ User │────→│  UI  │────→│ PassiAgent │────→│ LLM  │────→│ Tool      │
│      │     │(TUI) │     │ (ReAct Loop) │     │Client│     │Registry   │
└──────┘     └──────┘     └──────────────┘     └──────┘     └───────────┘
    │           │               │                   │              │
    │  1. input │  2. wire.emit │  3. build context │  4. chat()   │
    │           │               │                   │              │
    │           │               │  5. tool_calls ◄──┘              │
    │           │               │                                   │
    │           │               │  6. execute(tool, params) ───────→│
    │           │               │  7. result ◄──────────────────────┘
    │           │               │                                   │
    │           │  8. response  │  9. wire.emit(agent_message)     │
    │ 10. show  │               │                                   │
    │◄──────────┘               │                                   │
```

### 4.2 Python/R Code Execution

```
PassiAgent
    │
    ├─ run_python(code) → PythonExecutor
    │   └─ subprocess: python -c <code> → {stdout, stderr, exit_code}
    │
    └─ run_r(code) → RExecutor
        ├─ Primary: rpy2 bridge (ro.r(code))
        │   └─ Direct Bioconductor access, pandas↔R dataframe conversion
        └─ Fallback: Rscript subprocess
            └─ Rscript --no-save temp.R → {stdout, stderr, exit_code}
```

---

## 5. Communication Patterns

### 5.1 Wire Protocol (Internal)

All components communicate via `WireEvent` messages on the Wire:

```json
{
  "id": "a1b2c3d4e5f6",
  "type": "tool_call",
  "timestamp": "2026-06-25T14:30:52.123Z",
  "session_id": "session_20260625_143000",
  "data": {
    "name": "parse_omics_data",
    "params": {"path": "counts.csv"}
  }
}
```

### 5.2 REST API (External, Reserved)

```
POST   /api/v1/sessions              Create session
POST   /api/v1/sessions/{id}/chat    Send message → response
GET    /api/v1/sessions/{id}         Session status
DELETE /api/v1/sessions/{id}         Close session
GET    /api/v1/sessions/{id}/files   Session files
POST   /api/v1/tools/{name}          Direct tool invocation
GET    /api/v1/knowledge/search      Search methods/formats
WS     /api/v1/ws/{session_id}       Real-time chat
```

---

## 6. Directory Structure

```
digitagent/
├── pyproject.toml                  # Package metadata & dependencies
├── CLAUDE.md                       # Guidance for Claude Code instances
├── README.md                       # Project overview
│
├── src/
│   ├── passi/
│   │   ├── main.py                 # CLI entry point (`passi` command)
│   │   ├── config.py               # Configuration (env, YAML, JSON)
│   │   │
│   │   ├── soul/                   # Agent protocol & implementations
│   │   │   ├── protocol.py         # Soul abstract interface
│   │   │   ├── passi_agent.py    # Main PassiAgent
│   │   │   └── subagents/          # Domain-specific sub-agents
│   │   │       ├── omics_expert.py
│   │   │       ├── stats_expert.py
│   │   │       └── multi_omics.py      # (Phase 4)
│   │   │
│   │   ├── tools/                  # Tool definitions & registry
│   │   │   ├── base.py             # CallableTool[ParamsT] base class
│   │   │   ├── registry.py         # ToolRegistry
│   │   │   ├── io_tools.py         # read_file, write_file, parse_omics_data
│   │   │   ├── exec_tools.py       # run_python, run_r
│   │   │   ├── qc_tools.py         # (Phase 2)
│   │   │   ├── transcriptomics_tools.py  # (Phase 3)
│   │   │   ├── clinical_tools.py         # (Phase 3)
│   │   │   ├── integration_tools.py      # (Phase 4)
│   │   │   └── viz_tools.py              # (Phase 2)
│   │   │
│   │   ├── executors/              # Code execution backends
│   │   │   ├── python_executor.py
│   │   │   ├── r_executor.py
│   │   │   └── sandbox.py          # Docker sandbox
│   │   │
│   │   ├── wire/                   # Communication protocol
│   │   │   ├── protocol.py         # Wire (JSON-RPC pub/sub)
│   │   │   └── persistence.py      # wire.jsonl replay/audit
│   │   │
│   │   ├── ui/                     # User interfaces
│   │   │   ├── cli.py              # Rich TUI (chat mode)
│   │   │   ├── print_mode.py       # Non-interactive batch mode
│   │   │   └── prompts.py          # System prompt templates
│   │   │
│   │   ├── api/                    # Web API (reserved)
│   │   │   ├── server.py           # FastAPI application
│   │   │   ├── routes.py           # API route handlers
│   │   │   └── schemas.py          # Pydantic request/response models
│   │   │
│   │   ├── knowledge/              # Domain knowledge base
│   │   │   ├── formats.py          # Data format registry (40+ formats)
│   │   │   ├── methods.py          # Statistical methods catalog (50+ methods)
│   │   │   └── pipelines.py        # Predefined analysis workflows
│   │   │
│   │   └── infra/                  # Infrastructure
│   │       ├── runtime.py          # DI container
│   │       ├── session.py          # Session management
│   │       ├── llm_client.py       # LLM provider abstraction
│   │       ├── context.py          # Context window management
│   │       └── provenance.py       # Provenance tracking
│   │
│   └── tests/
│       ├── test_soul/
│       ├── test_tools/
│       ├── test_executors/
│       └── test_integration/
│
├── environments/                   # Conda environment specs
│   ├── bioinfo-py.yml
│   └── bioinfo-r.yml
│
├── pipelines/                      # User-defined workflow definitions
│   ├── rnaseq_diff_expr.yaml
│   ├── multi_omics_mofa.yaml
│   └── survival_predictor.yaml
│
└── docs/
    ├── architecture.md             # This document
    ├── specification.md            # Software requirements spec
    └── design.md                   # Design principles & conventions
```

---

## 7. Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.11+ | Orchestration, tools, UI, API |
| **R Bridge** | rpy2 3.5+ / Rscript | Bioconductor method execution |
| **LLM** | Anthropic SDK, OpenAI SDK, Ollama | Multi-provider LLM abstraction |
| **CLI** | Rich 13+, Textual 0.40+ | Interactive terminal UI |
| **Web API** | FastAPI 0.110+, uvicorn, websockets | REST + WebSocket API |
| **Data** | pandas 2.0+, numpy, pyarrow, h5py | Data manipulation |
| **Bioinfo Python** | scanpy, anndata, gseapy | Single-cell, enrichment |
| **ML** | scikit-learn, xgboost, shap | Machine learning integration |
| **Config** | pydantic-settings, pyyaml | Configuration management |
| **Code Quality** | ruff, mypy, pytest | Linting, type checking, testing |
| **Sandbox** | Docker (optional) | Full reproducibility |

---

## 8. Key Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Python-first, R via rpy2/subprocess** | Python for orchestration; R for mature Bioconductor methods. Avoids dual-language agent loop complexity. |
| 2 | **Pydantic schema enforcement** | Parameter validity: 20% (freeform code) → 98% (schema-constrained). Validated by ChatSpatial. |
| 3 | **Tool-first, not code-gen-first** | Domain expertise in versioned tool specs; LLM selects and orchestrates tools, does not improvise analysis logic. |
| 4 | **Wire protocol for UI agnosticism** | All communication through JSON-RPC events; CLI, Web, and SDK share the same backend. |
| 5 | **Sub-agents with context isolation** | Only analysis results flow back; prevents context pollution from intermediate steps. |
| 6 | **Session-as-directory** | All session state in a single folder: wire log, checkpoints, data, metadata. Fully portable and auditable. |
| 7 | **Lazy DI initialization** | Services created on first access; enables fast cold start for simple commands. |

---

## 9. Test Architecture (V-Model Test Layers)

The test architecture mirrors the system architecture layers, following the V-Model principle that each design layer has a corresponding test layer.

### 9.1 Test Pyramid

```
         ┌─────────┐
         │   UAT   │  ← 验收测试: 端到端用户场景
         │  5-10   │     (specification.md 验收标准)
         ├─────────┤
         │ System  │  ← 系统测试: CLI/API 完整链路
         │ 20-30   │     (接口规格验证)
         ├─────────┤
         │Integrat.│  ← 集成测试: 组件间交互
         │ 50-80   │     (Agent↔ToolRegistry, Wire↔Persistence)
         ├─────────┤
         │  Unit   │  ← 单元测试: 函数/类/工具
         │ 200+    │     (每个公开方法 ≥4 个测试)
         └─────────┘
```

### 9.2 V-Model Test Layer Mapping

| V-Model Stage | Test Layer | Scope | Tools | Location |
|--------------|-----------|-------|-------|----------|
| **Module Design** | Unit Test | Single function/class/tool | pytest, pytest-asyncio | `tests/test_tools/`, `tests/test_infra/`, `tests/test_executors/` |
| **Architecture Design** | Integration Test | Cross-component: Agent↔Tool, Wire↔Persistence | pytest + fixtures | `tests/test_integration/` |
| **System Design** | System Test | Full CLI/API command chain | pytest + subprocess | `tests/test_integration/` |
| **Requirements Analysis** | Acceptance Test | End-to-end bioinformatics analysis scenarios | pytest + test data | `tests/test_integration/` |

### 9.3 Test Fixture Architecture

```
tests/
├── conftest.py                          # Root fixtures
│   ├── test_config                      # Shared PassiConfig (session scope)
│   ├── temp_session_dir                 # Isolated session directory (function scope)
│   └── runtime                          # DI container with test config (module scope)
│
├── fixtures/                            # Test data factories (NOT tests)
│   ├── data_factories.py                # build_csv(), build_vcf(), build_count_matrix()
│   └── mock_llm.py                      # FakeLLMClient, StubToolRegistry
│
├── test_tools/                          # Unit tests: Tool layer
│   ├── conftest.py                      # tool_registry fixture
│   ├── test_io_tools.py                 # ReadFileTool, WriteFileTool, ParseOmicsDataTool
│   └── test_exec_tools.py               # RunPythonTool, RunRTool
│
├── test_executors/                      # Unit tests: Execution layer
│   └── test_python_executor.py          # PythonExecutor with sandbox
│
├── test_infra/                          # Unit tests: Infrastructure layer
│   ├── test_config.py                   # Config loading, env override
│   ├── test_session.py                  # Session CRUD, checkpoint
│   ├── test_context.py                  # Context add, compact, clear
│   └── test_llm_client.py               # Anthropic/OpenAI/Ollama client (mocked)
│
├── test_wire/                           # Unit tests: Communication layer
│   └── test_protocol.py                 # Wire publish, subscribe, persist
│
├── test_knowledge/                      # Unit tests: Knowledge layer
│   ├── test_formats.py                  # Format detection, domain lookup
│   └── test_methods.py                  # Method search, domain filtering
│
├── test_soul/                           # Unit tests: Agent layer
│   └── test_passi_agent.py            # chat(), execute_tool(), stream (mocked LLM)
│
└── test_integration/                    # Integration + System + Acceptance tests
    ├── test_agent_tool_roundtrip.py     # User message → Agent → Tool → Result
    ├── test_wire_persistence.py         # Wire events → wire.jsonl → replay
    ├── test_session_persistence.py      # Session create → save → load → restore
    ├── test_rna_seq_workflow.py         # Acceptance: RNA-seq diff expr E2E
    └── test_survival_analysis.py        # Acceptance: Survival analysis E2E
```

### 9.4 Test Data Strategy

| Data Type | Source | Purpose |
|-----------|--------|---------|
| **Synthetic minimal** | `fixtures/data_factories.py` | Unit tests — tiny dataframes (5×3), controlled edge cases |
| **Synthetic realistic** | `fixtures/data_factories.py` | Integration tests — realistic dimensions (100×20), internal structure |
| **Public datasets** | TCGA, GEO, MetaboLights, PRIDE subsamples | Acceptance tests — real bioinformatics data, validates tool correctness |
| **Mock LLM responses** | `fixtures/mock_llm.py` | Agent tests — FakeLLMClient returns predefined tool call sequences |

---

## 10. Implementation Phases

| Phase | Scope | Milestone |
|-------|-------|-----------|
| **Phase 1** | Core infrastructure: config, runtime, session, context, LLM client, Wire protocol, Soul/PassiAgent, tool registry, I/O + exec tools, CLI TUI | `passi chat` functional |
| **Phase 2** | Knowledge layer: format auto-detection, methods catalog, pipelines, QC tools, visualization tools | Data-aware agent |
| **Phase 3** | Single-omics tools: transcriptomics (priority), genomics, epigenetics, proteomics, metabolomics, clinical stats | Full single-omics analysis |
| **Phase 4** | Multi-omics integration: MOFA, DIABLO, SNF, ML integration, advanced visualization | Multi-omics analysis |
| **Phase 5** | Web API, WebSocket, batch/script mode, session export, Docker sandbox, documentation | Production release v1.0 |
