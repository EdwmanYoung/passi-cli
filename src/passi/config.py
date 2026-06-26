"""Configuration management for PassiAgent.

Loads settings from environment variables, .env files, and config files.
Config priority (lowest to highest):
  1. Code defaults
  2. ~/.passi/settings.yaml (user-global baseline)
  3. Project .passi/settings.yaml (overrides user-global)
  4. CWD .env file (project-specific)
  5. CLI --config YAML/JSON file (explicit, overrides .env)
  6. PASSI_* environment variables
  7. PASSI_CONFIG JSON env override
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Passi home directory ──────────────────────────────────────────────────


def _passi_home() -> Path:
    """Return the Passi user config directory (cross-project)."""
    return Path.home() / ".passi"


def _project_passi_dir(start: Path | None = None) -> Path | None:
    """Walk up from `start` (default: CWD) to find the nearest `.passi/` directory.

    Returns the `.passi/` directory path if found, None otherwise.
    Stops at the filesystem root.
    """
    current = (start or Path.cwd()).resolve()
    root = current.anchor  # e.g. "C:\\" or "/"
    while str(current) != root:
        candidate = current / ".passi"
        if candidate.is_dir():
            return candidate
        current = current.parent
    # Check root
    candidate = Path(root) / ".passi"
    return candidate if candidate.is_dir() else None


def _resolve_passi_dir() -> Path:
    """Return the active passi config directory.

    Priority: project `.passi/` > `~/.passi/`.
    Creates `~/.passi/` as fallback if neither exists.
    """
    project = _project_passi_dir()
    if project is not None:
        return project
    return _passi_home()


SETTINGS_TEMPLATE = """\
# =============================================================================
#  PassiAgent — User Settings
# =============================================================================
#  Location: ~/.passi/settings.yaml (global) or <project>/.passi/settings.yaml
#  Project .passi/settings.yaml takes precedence over the global one.
#
#  Override order (highest wins):
#    1. PASSI_CONFIG JSON environment variable
#    2. PASSI_* environment variables (e.g. PASSI_ANTHROPIC__API_KEY)
#    3. CLI --config <file> (YAML or JSON)
#    4. CWD .env file (project-specific)
#    5. Project .passi/settings.yaml
#    6. ~/.passi/settings.yaml (this file, user-global)
#    7. Code defaults
# =============================================================================

# Default LLM provider: "anthropic", "openai", or "ollama"
default_provider: anthropic

# ── LLM Providers ──────────────────────────────────────────────────────
#  API keys can also be set via environment variables:
#    ANTHROPIC_API_KEY, OPENAI_API_KEY
#  Base URLs can point to compatible APIs (DeepSeek, OpenRouter, etc.)

anthropic:
  # Your API key (or set ANTHROPIC_API_KEY env var)
  api_key: ""
  # API endpoint — defaults shown; change for proxies or compatible APIs
  base_url: https://api.deepseek.com/anthropic
  # Model name: claude-sonnet-4-6, claude-opus-4-7, deepseek-v4-pro, etc.
  model: deepseek-v4-pro
  # Max tokens per LLM response (not per tool-call iteration)
  max_tokens: 8192
  # Temperature: 0.0 = deterministic, higher = more creative
  temperature: 0.0
  # Extended thinking budget in tokens (0 = disabled)
  thinking_budget_tokens: 0

openai:
  api_key: ""
  base_url: https://api.deepseek.com
  model: deepseek-v4-pro
  max_tokens: 4096
  temperature: 0.0

# ── Code Execution ─────────────────────────────────────────────────────
#  Controls how Python and R code are executed during analysis.
#  R auto-detection order (when r_home is empty):
#    1) PASSI_R_HOME / R_HOME environment variables
#    2) Project-local ./R/ directory (recommended for rpy2)
#    3) System R under Program Files\\R (newest version first)

execution:
  # Python interpreter to use.
  #   "python"         → system default Python
  #   "C:/path/python" → specific Python/conda environment
  #   "python3.11"     → version-specific executable
  python_path: python

  # R home directory — path to the R installation root folder.
  # Examples:
  #   Windows: C:/Program Files/R/R-4.4.0
  #   Linux:   /usr/lib/R
  #   macOS:   /Library/Frameworks/R.framework/Resources
  # Auto-detected on first run — set manually only if detection fails.
  r_home: "$r_home_detected"

  # R library path — where R packages are installed.
  # Defaults to .passi/R-lib/ in the project (or ~/.passi/R-lib/).
  # This keeps project R packages isolated from system R packages.
  # Set to a custom path if you share an R library across projects.
  r_lib_path: ""

  # Use rpy2 bridge for Python↔R integration (recommended).
  # When disabled, all R code runs via Rscript subprocess.
  rpy2_enabled: true

  # Maximum time (seconds) a single code execution can run.
  # Increase for large datasets or complex models.
  timeout_seconds: 300

  # Maximum captured stdout/stderr size in bytes (10 MB default).
  # Output beyond this limit is truncated before sending to the LLM.
  # Full logs are always preserved on disk.
  max_output_bytes: 10485760

# ── Plan Mode ──────────────────────────────────────────────────────────
#  Controls the interactive plan creation and execution workflow.

plan:
  # Minimum number of clarifying questions before plan generation (3-4)
  qa_min_questions: 3
  qa_max_questions: 4
  # Maximum times a plan can be rejected and revised
  max_recycles: 3
"""


def _ensure_passi_home() -> Path:
    """Create the active .passi/ directory with sessions/ and settings template.

    Priority: project `.passi/` > `~/.passi/`. Idempotent — safe to call on every startup.
    Auto-detects R installation and pre-fills the settings template.
    """
    passi_dir = _resolve_passi_dir()
    (passi_dir / "sessions").mkdir(parents=True, exist_ok=True)
    settings_file = passi_dir / "settings.yaml"
    if not settings_file.exists():
        r_home = _find_r_home("")
        r_lib = str(passi_dir / "R-lib")
        template = SETTINGS_TEMPLATE.replace("$r_home_detected", r_home or "")
        if r_home:
            template = template.replace(
                "# Auto-detected on first run — set manually only if detection fails.",
                "# Auto-detected on first run."
            )
        settings_file.write_text(template, encoding="utf-8")
        # Ensure R-lib directory exists
        (passi_dir / "R-lib").mkdir(parents=True, exist_ok=True)
    return passi_dir


class LLMProviderConfig(BaseSettings):
    """Configuration for a single LLM provider."""

    api_key: str = ""
    model: str = ""
    base_url: Optional[str] = None
    max_tokens: int = 4096
    tool_call_max_tokens: int = 4096  # token budget for tool-calling iterations
    temperature: float = 0.0
    enabled: bool = True


class AnthropicConfig(LLMProviderConfig):
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 16384
    thinking_budget_tokens: int = 0  # 0 disables extended thinking


class OpenAIConfig(LLMProviderConfig):
    model: str = "gpt-4o"
    base_url: Optional[str] = None


class OllamaConfig(LLMProviderConfig):
    model: str = "llama3.1"
    base_url: str = "http://localhost:11434/v1"
    enabled: bool = False


class ExecutionConfig(BaseSettings):
    """Code execution sandbox configuration.

    R is auto-detected in this order:
      1) Explicit r_home config / PASSI_EXECUTION__R_HOME env var
      2) PASSI_R_HOME / R_HOME environment variables
      3) Project-local ``./R/`` directory (recommended for rpy2)
      4) System R under ``Program Files/R``
    """

    python_path: str = "python"
    r_path: str = "Rscript"
    r_home: str = ""  # R home — auto-detected if empty
    r_lib_path: str = ""  # custom R library — auto-set to ./R-lib if empty
    rpy2_enabled: bool = True
    timeout_seconds: int = 300
    max_output_bytes: int = 10 * 1024 * 1024  # 10 MB
    conda_env: Optional[str] = None
    docker_image: Optional[str] = None

    @field_validator("r_home", mode="before")
    @classmethod
    def resolve_r_home(cls, v: str | Path) -> str:
        return _find_r_home(v)

    @field_validator("r_lib_path", mode="before")
    @classmethod
    def resolve_r_lib_path(cls, v: str) -> str:
        if v:
            return str(v)
        # Default: .passi/R-lib/ in the active passi config directory
        passi_dir = _resolve_passi_dir()
        default = passi_dir / "R-lib"
        default.mkdir(parents=True, exist_ok=True)
        return str(default.resolve())

    @property
    def r_binary(self) -> str:
        """Path to R executable (R.exe / R)."""
        if self.r_home:
            home = Path(self.r_home)
            for subpath in ("bin/R.exe", "bin/R"):
                exe = home / subpath
                if exe.exists():
                    return str(exe)
        return self.r_path

    @property
    def rscript_binary(self) -> str:
        """Path to Rscript executable."""
        if self.r_home:
            home = Path(self.r_home)
            for subpath in ("bin/Rscript.exe", "bin/Rscript"):
                exe = home / subpath
                if exe.exists():
                    return str(exe)
        return self.r_path


def _find_r_home(explicit: str | Path = "") -> str:
    """Auto-detect R installation.

    Resolution order:
      1) Explicit value (from config file or PASSI_EXECUTION__R_HOME)
      2) PASSI_R_HOME / R_HOME env vars
      3) Project-local ./R/ directory
      4) System R under Program Files\\R (newest first)
    """
    # 1) Explicit — trust user if provided, even if not a valid R home
    if explicit:
        path = Path(explicit)
        if _is_r_home(path):
            return str(path.resolve())
        # Maybe it's a version number or partial path relative to project
        resolved = Path.cwd() / path
        if _is_r_home(resolved):
            return str(resolved.resolve())
        # Not a valid R install, but user explicitly set it — return as-is
        return str(path)

    # 2) Environment variables
    for env_var in ("PASSI_R_HOME", "R_HOME"):
        env_val = os.environ.get(env_var, "")
        if env_val:
            p = Path(env_val)
            if _is_r_home(p):
                return str(p.resolve())

    # 3) Project-local R
    local_r = Path.cwd() / "R"
    if _is_r_home(local_r):
        return str(local_r.resolve())

    # 4) System R (Program Files\\R\\R-x.x.x, newest first)
    program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
    r_base = Path(program_files) / "R"
    if r_base.exists():
        versions = sorted(
            [d for d in r_base.iterdir() if d.is_dir() and _is_r_home(d)],
            reverse=True,
        )
        if versions:
            return str(versions[0].resolve())

    return str(explicit) if explicit else ""


def _is_r_home(path: Path) -> bool:
    """Check if a path looks like a valid R home directory."""
    return (
        (path / "bin" / "R.exe").exists()
        or (path / "bin" / "R").exists()
    )


def _convert_env_value(v: str) -> Any:
    """Convert a .env string value to its likely Python type."""
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    if v.isdigit():
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v


def _env_to_nested(env_path: Path, prefix: str = "PASSI_") -> dict[str, Any]:
    """Parse a .env file and convert PASSI_* flat keys to nested dict.

    Example: PASSI_ANTHROPIC__API_KEY=sk-xxx -> {"anthropic": {"api_key": "sk-xxx"}}
    """
    from dotenv import dotenv_values

    flat = dotenv_values(str(env_path))
    result: dict[str, Any] = {}
    for key, value in flat.items():
        if value is None or not key.startswith(prefix):
            continue
        sub_key = key[len(prefix):].lower()
        parts = sub_key.split("__")
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = _convert_env_value(str(value))
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two nested dicts, with override taking precedence."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class SessionConfig(BaseSettings):
    """Session management configuration."""

    sessions_dir: Path = Field(default_factory=lambda: _resolve_passi_dir() / "sessions")
    max_sessions: int = 100
    checkpoint_interval: int = 5  # messages between auto-checkpoints
    wire_file: str = "wire.jsonl"


class HooksConfig(BaseSettings):
    """User hook configuration — persisted to .passi/hooks.yaml."""

    hooks_file: Path = Field(default_factory=lambda: _resolve_passi_dir() / "hooks.yaml")
    enabled: bool = True  # master kill switch for all hooks
    timeout_seconds: int = 30  # per-hook execution timeout


class PlanConfig(BaseSettings):
    """Plan mode interaction configuration."""

    qa_min_questions: int = 3  # minimum clarifying questions before plan creation
    qa_max_questions: int = 4  # maximum clarifying questions
    max_recycles: int = 3  # maximum plan reject/recycle iterations


class PassiConfig(BaseSettings):
    """Root configuration for PassiAgent."""

    model_config = SettingsConfigDict(
        env_prefix="PASSI_",
        env_nested_delimiter="__",
        extra="allow",
    )

    # LLM providers
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)

    # Default provider
    default_provider: str = "anthropic"

    # Execution
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    # Session
    session: SessionConfig = Field(default_factory=SessionConfig)

    # General
    debug: bool = False
    data_dir: Path = Path("./data")
    result_dir: Path = Path("./result")  # primary output directory for analysis results

    # Hooks
    hooks: HooksConfig = Field(default_factory=HooksConfig)

    # Plan mode
    plan: PlanConfig = Field(default_factory=PlanConfig)

    # Prompt templating
    prompt_template_dir: str = ""  # "" = use built-in defaults from passi/prompts/
    enable_data_format_check: bool = True  # append data format check instructions to system prompt
    afk_mode: bool = False  # AFK autonomous mode: auto-plan, auto-execute, never ask user

    @field_validator("data_dir", "result_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()

    @property
    def output_dir(self) -> Path:
        """Deprecated: use result_dir instead."""
        import warnings
        warnings.warn(
            "PassiConfig.output_dir is deprecated. Use result_dir instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.result_dir

    def get_llm_config(self, provider: str | None = None) -> LLMProviderConfig:
        """Get LLM config for a specific provider."""
        provider = provider or self.default_provider
        configs = {
            "anthropic": self.anthropic,
            "openai": self.openai,
            "ollama": self.ollama,
        }
        if provider not in configs:
            msg = f"Unknown LLM provider: {provider}. Choose from: {list(configs.keys())}"
            raise ValueError(msg)
        return configs[provider]


def load_config(config_path: str | Path | None = None) -> PassiConfig:
    """Load configuration with layered priority.

    Layers (lowest to highest):
      1. Code defaults
      2. ~/.passi/settings.yaml (user-global baseline)
      3. Project .passi/settings.yaml (overrides user-global)
      4. CWD .env file (project-specific)
      5. CLI --config YAML/JSON file (explicit, overrides .env)
      6. PASSI_* environment variables
      7. PASSI_CONFIG JSON env override
    """
    _ensure_passi_home()

    file_data: dict = {}

    # Layer 2: ~/.passi/settings.yaml (user-global baseline)
    global_settings = _passi_home() / "settings.yaml"
    if global_settings.exists():
        data = yaml.safe_load(global_settings.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            file_data = _deep_merge(file_data, data)

    # Layer 3: Project .passi/settings.yaml (overrides global)
    project = _project_passi_dir()
    if project is not None:
        project_settings = project / "settings.yaml"
        if project_settings.exists():
            data = yaml.safe_load(project_settings.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                file_data = _deep_merge(file_data, data)

    # Layer 4: CWD .env (project-specific, overrides settings files)
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        env_data = _env_to_nested(cwd_env)
        file_data = _deep_merge(file_data, env_data)

    # Layer 5: CLI --config YAML/JSON file (explicit, overrides .env)
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    file_data = _deep_merge(file_data, data)
            elif path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    file_data = _deep_merge(file_data, data)

    # Create config — env vars (Layer 6) override all via pydantic-settings env_prefix
    config = PassiConfig(**file_data) if file_data else PassiConfig()

    # Layer 7: PASSI_CONFIG JSON env override (highest priority)
    env_override = os.environ.get("PASSI_CONFIG", "")
    if env_override:
        override = json.loads(env_override)
        config = PassiConfig(**{**config.model_dump(), **override})
    return config
