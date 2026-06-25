"""Configuration management for PassiAgent.

Loads settings from environment variables, .env files, and config files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        # Default: project-local R library
        default = Path.cwd() / "R-lib"
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


class SessionConfig(BaseSettings):
    """Session management configuration."""

    sessions_dir: Path = Path("./sessions")
    max_sessions: int = 100
    checkpoint_interval: int = 5  # messages between auto-checkpoints
    wire_file: str = "wire.jsonl"


class PassiConfig(BaseSettings):
    """Root configuration for PassiAgent."""

    model_config = SettingsConfigDict(
        env_prefix="PASSI_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
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
    output_dir: Path = Path("./output")

    # Prompt templating
    prompt_template_dir: str = ""  # "" = use built-in defaults from passi/prompts/
    enable_data_format_check: bool = True  # append data format check instructions to system prompt
    afk_mode: bool = False  # AFK autonomous mode: auto-plan, auto-execute, never ask user

    @field_validator("data_dir", "output_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()

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
    """Load configuration from a YAML/JSON file and environment variables.

    Environment variables take precedence over file settings.
    """
    file_data: dict = {}
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            pass  # Silently skip — use defaults + env vars
        elif path.suffix in (".yaml", ".yml"):
            import yaml

            with open(path) as f:
                data = yaml.safe_load(f)
            if data:
                file_data = data
        elif path.suffix == ".json":
            import json

            with open(path) as f:
                data = json.load(f)
            if data:
                file_data = data

    # Start with defaults, apply file data, env vars naturally take precedence
    config = PassiConfig(**file_data) if file_data else PassiConfig()

    # Apply environment variable overrides
    env_override = os.environ.get("PASSI_CONFIG", "")
    if env_override:
        import json

        override = json.loads(env_override)
        config = PassiConfig(**{**config.model_dump(), **override})
    return config
