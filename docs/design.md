# PassiAgent — Design Principles & Conventions

> **Version:** 0.1.0 | **Date:** 2026-06-25

## 1. Core Design Principles

### 1.1 统一数据模型
所有组学数据继承统一的 `OmicsDataset` 抽象，保证跨组学接口一致性。不论输入是转录组 count matrix 还是代谢物丰度表，API 保持一致。

### 1.2 双语言支持
Python 为主编排语言，R 为计算方法桥接。关键方法（DESeq2, limma, WGCNA, MOFA2, mixOmics）优先用 R/Bioconductor 实现，通过 rpy2 桥接或 Rscript 子进程调用。Python 负责可视化、ML 和通用计算。

### 1.3 方法注册表
所有分析方法通过 `knowledge/methods.py` 注册表管理，包含方法名、所属领域、执行后端（Python/R）、所需包、输入输出类型。新增分析方法只需在注册表中添加条目。

### 1.4 会话上下文
维护分析状态，支持增量式分析。用户可以逐步深入：先做差异分析 → 再做富集分析 → 再做网络分析，全程保持数据上下文。

### 1.5 品牌一致性
CLI 输出、图表主题统一使用蓝白渐变品牌色。配色方案：
- **主色:** `#2563EB` (蓝) / `#3B82F6`
- **辅色:** `#F8FAFC` (白) / `#1E293B` (深色)
- **成功:** `#10B981` | **警告:** `#F59E0B` | **错误:** `#EF4444`

### 1.6 渐进式暴露
功能接口按复杂度分层暴露：
- **CLI 命令** → 最直接，单次调用
- **REPL 对话** → 交互式，有上下文
- **REST API** → 编程调用，无状态
- **Python SDK** → 深度集成，全功能

---

## 2. Code Conventions

### 2.1 Python 规范

- Python 3.11+，使用现代类型注解（`list[dict]` 而非 `List[Dict]`）
- 所有公开函数必须有类型注解
- 使用 `from __future__ import annotations` 延迟求值
- 单行 ≤100 字符（ruff line-length=100）
- 使用 ruff 格式化 + lint，mypy 类型检查

### 2.2 项目组织

```
每个模块包含:
├── __init__.py        # 公开 API 导出
├── <module>.py        # 核心实现（一个模块一个主文件）
└── ...                # 辅助模块（如有必要）
```

原则：
- 一个类一个主文件（除非紧密耦合）
- Pydantic 模型优先：参数、配置、消息均用 Pydantic 建模
- 异步优先：所有 I/O 操作用 async/await
- 无全局状态：通过 Runtime DI 容器传递依赖

### 2.3 命名约定

| 类型 | 约定 | 示例 |
|------|------|------|
| 类名 | PascalCase | `PassiAgent`, `ToolRegistry` |
| 函数/方法 | snake_case | `parse_omics_data()`, `get_schemas()` |
| 常量 | UPPER_SNAKE | `SYSTEM_PROMPT`, `DEFAULT_MAX_MESSAGES` |
| 私有成员 | _prefix | `_llm_client`, `_compaction_index` |
| 模块文件 | snake_case | `llm_client.py`, `io_tools.py` |

### 2.4 错误处理

- 工具执行错误返回 `{"success": False, "error": "message"}` 而非抛异常
- 基础设施错误（LLM API, 文件不存在）可抛异常，由上层捕获
- 所有用户可见错误信息使用中文（可切换到英文）

---

## 3. API Design

### 3.1 工具接口

每个工具遵循统一接口：

```python
class SomeTool(CallableTool[SomeParams]):
    name = "tool_name"              # 唯一标识，snake_case
    description = "What it does"    # LLM 选择工具的依据
    params_model = SomeParams       # Pydantic model
```

工具返回格式：

```python
{
    "success": True,                # bool: 是否成功
    "result": ...,                  # Any: 主要结果
    "error": "",                    # str: 错误信息（success=False 时）
    "files": [...],                 # list[str]: 输出文件路径
    "figures": [...],               # list[str]: 输出图片路径
}
```

### 3.2 Wire 事件

所有消息通信遵循 JSON-RPC 风格的事件模型：

```json
{
    "id": "uuid-v4-12chars",
    "type": "event_type_name",
    "timestamp": "ISO-8601 UTC",
    "session_id": "session_id",
    "data": {},
    "metadata": {}
}
```

---

## 4. Tool Design Guidelines

### 4.1 何时创建新工具

- 该操作需要被 LLM 调用（离散的、定义明确的分析步骤）
- 该操作有明确的输入/输出契约
- 该操作可能被多个分析流程复用

### 4.2 何时使用 run_python / run_r

- 一次性的、用户特定的数据处理
- 分析流程中的胶水代码
- 没有现成工具的探索性分析

### 4.3 工具描述质量

工具的 `description` 是 LLM 选择工具的唯一依据。好的描述：
- 说清楚做什么（不是怎么实现）
- 说明适用场景（"用于 RNA-seq 差异表达分析"）
- 提及关键参数（"支持 DESeq2, edgeR, limma 三种方法"）

---

## 5. LLM Interaction Patterns

### 5.1 系统提示设计

系统提示是 agent 行为的核心约束。层次结构：
- **角色定义** — 你是谁，能做什么
- **能力列表** — 具体的分析能力边界
- **行为准则** — 质量要求和工作流程
- **约束条件** — 不能做什么

### 5.2 ReAct 循环参数

- 最大迭代次数: 20（单次对话）
- 上下文压缩阈值: 80,000 tokens
- 检查点间隔: 每 5 条消息

### 5.3 子代理委托

子代理在隔离上下文中运行，仅结果回传主代理：
- **OmicsExpert** — 单组学领域分析
- **StatsExpert** — 临床统计和生物统计
- **MultiOmicsIntegrator** (Phase 4) — 多组学整合分析

---

## 6. TDD — 测试驱动开发 (Test-Driven Development)

本项目严格采用 TDD 方法进行编码开发，遵循 **Red-Green-Refactor** 循环。

### 6.1 TDD 核心循环

```
┌──────────────────────────────────────────┐
│                                          │
│  ① RED     ② GREEN     ③ REFACTOR      │
│  (写失败    (最少代码    (重构代码       │
│   的测试)   让测试通过)   优化设计)       │
│                                          │
│  Write a   Make it     Refactor         │
│  failing   pass with   without          │
│  test →    minimal  →  changing         │
│            code         behavior ──→ 循环   │
│                                          │
└──────────────────────────────────────────┘
```

**Step 1 — RED (写失败的测试):**
1. 根据模块设计规格，编写测试用例
2. 运行测试，确认测试 FAIL（红）
3. 验证测试用例本身正确（不是因测试 bug 失败）

**Step 2 — GREEN (最少代码让测试通过):**
1. 编写刚好足够让测试通过的代码
2. 不考虑优化、不考虑边缘情况、不考虑美观
3. 运行测试，确认全部 PASS（绿）
4. 如果 30s 内不能让测试通过 → 回退，重新思考

**Step 3 — REFACTOR (重构优化):**
1. 消除重复代码
2. 改善命名和结构
3. 运行测试确认仍然全绿
4. 不改变外部行为

### 6.2 TDD 三定律

| 定律 | 描述 |
|------|------|
| **第一定律** | 在编写不能通过的单元测试前，不可编写生产代码 |
| **第二定律** | 只编写刚好无法通过的单元测试（编译失败也算不通过） |
| **第三定律** | 只编写刚好足以让当前失败测试通过的生产代码 |

### 6.3 测试文件命名与组织规范

```
src/
├── passi/
│   ├── tools/
│   │   ├── io_tools.py          # 生产代码
│   │   ├── exec_tools.py
│   │   └── ...
│   └── infra/
│       ├── session.py
│       └── ...
└── tests/
    ├── conftest.py               # 全局 fixtures
    ├── fixtures/                  # 测试数据工厂
    │   ├── __init__.py
    │   ├── data_factories.py     # 模拟数据生成器
    │   └── mock_llm.py           # LLM mock 工具
    ├── test_tools/
    │   ├── __init__.py
    │   ├── test_io_tools.py      # 对应 tools/io_tools.py
    │   ├── test_exec_tools.py
    │   └── test_registry.py
    ├── test_executors/
    │   ├── test_python_executor.py
    │   └── test_r_executor.py
    ├── test_infra/
    │   ├── test_session.py
    │   ├── test_context.py
    │   ├── test_llm_client.py
    │   └── test_provenance.py
    ├── test_soul/
    │   ├── test_protocol.py
    │   └── test_passi_agent.py
    ├── test_wire/
    │   └── test_protocol.py
    └── test_integration/
        ├── test_agent_tool_roundtrip.py
        └── test_session_persistence.py
```

命名规则:
- 测试文件: `test_<module_name>.py`
- 测试类: `Test<ClassName>`
- 测试函数: `test_<what>_<condition>_<expected>`
- 示例: `test_parse_omics_data_with_csv_returns_count_matrix()`

### 6.4 Fixtures 与 Mock 策略

#### Fixtures 层次

```python
# conftest.py — 全局 fixtures
# session scope: 整个测试会话共享（慢初始化）
# module scope: 模块内所有测试共享
# function scope: 每个测试函数独立（默认）

@pytest.fixture(scope="session")
def test_config():          # 共享的 PassiConfig
    ...

@pytest.fixture(scope="function")
def temp_session(tmp_path): # 每个测试独立的会话
    ...

@pytest.fixture(scope="module")
def sample_count_matrix():  # 模块内共享的测试数据
    ...
```

#### Mock 策略优先级

1. **优先使用真实对象** — 纯函数、无副作用的代码不 mock
2. **Fake 实现** — 用内存版替代外部依赖（如 FakeLLMClient）
3. **pytest-mock** — 对 I/O 边界进行 mock（网络、文件系统）
4. **Monkeypatch** — 对模块级变量进行替换

#### Mock 边界规则

| 需要 Mock | 不需要 Mock |
|-----------|------------|
| LLM API 调用 (anthropic, openai) | Pydantic 模型校验 |
| R/rpy2 子进程调用 | 数据格式解析 |
| 文件系统 I/O（大量文件场景） | 工具注册/查找 |
| 网络请求 (httpx) | DI 容器装配 |
| Docker 执行 | 会话元数据读写 |

### 6.5 TDD 编码规范

#### 每个测试的结构：AAA 模式

```python
async def test_read_file_with_csv_returns_preview():
    # Arrange — 准备测试数据和环境
    tool = ReadFileTool()
    csv_path = create_test_csv(rows=100, cols=5)
    params = ReadFileParams(path=str(csv_path))

    # Act — 执行被测代码
    result = await tool.execute(params)

    # Assert — 验证结果
    assert result["success"] is True
    assert result["format"] == "csv"
    assert result["shape"] == [100, 5]
    assert len(result["preview"]) == 100
```

#### 测试命名规范

```
test_<被测方法>_<条件/场景>_<预期结果>

示例:
test_parse_omics_data_with_vcf_returns_variant_count
test_execute_tool_with_unknown_name_returns_error
test_session_load_with_nonexistent_id_raises_not_found
test_chat_with_tool_calls_executes_tools_in_order
```

#### 每个工具的测试清单

| # | 测试类型 | 最少数量 | 示例 |
|---|---------|---------|------|
| 1 | Happy path — 正常输入 | 1 | 正确 CSV → 返回预览 |
| 2 | Error — 无效输入 | 1 | 文件不存在 → 返回 error |
| 3 | Edge — 边界条件 | 1 | 空文件 / 超大参数 |
| 4 | Schema — 参数验证 | 1 | 缺少必填字段 → Pydantic 抛异常 |

### 6.6 TDD 工作流示例

以开发 `normalize_data` 工具为例:

```
# ── RED: 先写测试 ──
# 文件: tests/test_tools/test_qc_tools.py

async def test_normalize_data_with_zscore_returns_normalized_matrix():
    tool = NormalizeDataTool()
    data = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]})
    params = NormalizeDataParams(method="zscore", data=data)
    result = await tool.execute(params)
    assert result["success"] is True
    # 验证标准化后均值≈0, 标准差≈1
    normalized = result["result"]
    for col in normalized.columns:
        assert abs(normalized[col].mean()) < 0.001
        assert abs(normalized[col].std() - 1.0) < 0.001
    # 测试 FAIL → 进入 GREEN

# ── GREEN: 最小实现 ──
# 文件: tools/qc_tools.py

class NormalizeDataTool(CallableTool[NormalizeDataParams]):
    name = "normalize_data"
    description = "Normalize omics data matrix"
    params_model = NormalizeDataParams

    async def execute(self, params):
        if params.method == "zscore":
            from scipy import stats
            result = params.data.apply(stats.zscore)
            return {"success": True, "result": result}
        return {"success": False, "error": f"Unknown method: {params.method}"}
    # 测试 PASS → 进入 REFACTOR

# ── REFACTOR: 优化实现 ──
# (提取方法、添加更优雅的错误处理...)
# 每次重构后运行测试，确保持续全绿
```

### 6.7 覆盖率目标

| 模块 | 行覆盖率 | 分支覆盖率 | 说明 |
|------|---------|-----------|------|
| `tools/*` | ≥ 90% | ≥ 85% | 工具层是可测试性最高的层 |
| `executors/*` | ≥ 85% | ≥ 80% | 需要 mock 子进程 |
| `infra/*` | ≥ 85% | ≥ 80% | 需要 mock LLM/文件系统 |
| `soul/*` | ≥ 80% | ≥ 75% | Agent 循环复杂度高 |
| `knowledge/*` | ≥ 90% | ≥ 85% | 纯数据，高覆盖率 |
| `wire/*` | ≥ 85% | ≥ 80% | 事件协议需要精确测试 |
| `ui/*` | ≥ 70% | ≥ 60% | UI 层部分依赖终端模拟 |
| **Overall** | **≥ 80%** | **≥ 75%** | — |

运行覆盖率:
```bash
pytest --cov=src/passi --cov-report=html --cov-report=term
```

---

## 7. Development Workflow

### 7.1 添加新工具 (TDD)

1. **RED** — 在 `tests/test_tools/` 创建测试文件, 写失败测试
2. **GREEN** — 在 `tools/` 下创建工具文件, 最小实现让测试通过
3. **REFACTOR** — 重构优化
4. 在 `passi_agent.py` 的 `_create_tool_registry()` 中注册
5. 在 `knowledge/methods.py` 中添加方法条目（如果是分析方法）

### 7.2 添加新组学领域 (TDD)

1. **RED** — 写数据格式检测测试 + 领域工具测试
2. **GREEN** — 添加格式条目 + 实现工具
3. 在 `knowledge/formats.py` 添加数据格式条目
4. 在 `knowledge/methods.py` 添加分析方法条目
5. 注册工具到 ToolRegistry
6. 更新系统提示模板
