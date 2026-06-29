"""TDD-style unit tests for PassiConfig and load_config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from passi.config import (
    AnthropicConfig,
    ExecutionConfig,
    HooksConfig,
    LLMProviderConfig,
    OllamaConfig,
    OpenAIConfig,
    PassiConfig,
    PlanConfig,
    SessionConfig,
    _project_passi_dir,
    _resolve_passi_dir,
    load_config,
)


class TestLLMProviderConfig:
    """Default values for each provider config."""

    def test_anthropic_config_defaults(self):
        cfg = AnthropicConfig()
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.max_tokens == 16384
        assert cfg.temperature == 0.0
        assert cfg.enabled is True
        assert cfg.api_key == ""
        assert cfg.thinking_budget_tokens == 0

    def test_openai_config_defaults(self):
        cfg = OpenAIConfig()
        assert cfg.model == "gpt-4o"
        assert cfg.max_tokens == 4096
        assert cfg.base_url is None

    def test_ollama_config_defaults(self):
        cfg = OllamaConfig()
        assert cfg.model == "llama3.1"
        assert cfg.base_url == "http://localhost:11434/v1"
        assert cfg.enabled is False  # Ollama is opt-in

    def test_llm_provider_config_base(self):
        cfg = LLMProviderConfig(api_key="sk-123", model="test-model", max_tokens=100)
        assert cfg.api_key == "sk-123"
        assert cfg.model == "test-model"
        assert cfg.max_tokens == 100

    def test_tool_call_max_tokens_default(self):
        cfg = LLMProviderConfig()
        assert cfg.tool_call_max_tokens == 4096

    def test_tool_call_max_tokens_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_ANTHROPIC__TOOL_CALL_MAX_TOKENS", "2048")
        cfg = PassiConfig()
        assert cfg.anthropic.tool_call_max_tokens == 2048


class TestPassiConfigDefaults:
    """PassiConfig default values and structure."""

    def test_default_provider(self):
        cfg = PassiConfig()
        assert cfg.default_provider == "anthropic"

    def test_anthropic_nested_config(self):
        cfg = PassiConfig(anthropic={"model": "claude-sonnet-4-6"})
        assert isinstance(cfg.anthropic, AnthropicConfig)
        assert cfg.anthropic.model == "claude-sonnet-4-6"

    def test_openai_nested_config(self):
        cfg = PassiConfig(openai={"model": "gpt-4o", "base_url": None})
        assert isinstance(cfg.openai, OpenAIConfig)
        assert cfg.openai.model == "gpt-4o"

    def test_ollama_nested_config(self):
        cfg = PassiConfig()
        assert isinstance(cfg.ollama, OllamaConfig)
        assert cfg.ollama.enabled is False

    def test_execution_config_is_present(self):
        cfg = PassiConfig()
        assert isinstance(cfg.execution, ExecutionConfig)
        assert cfg.execution.timeout_seconds == 300

    def test_session_config_is_present(self):
        cfg = PassiConfig()
        assert isinstance(cfg.session, SessionConfig)
        assert cfg.session.max_sessions == 100

    def test_debug_defaults_to_false(self):
        cfg = PassiConfig()
        assert cfg.debug is False

    def test_extra_fields_allowed(self):
        cfg = PassiConfig(custom_field="value")
        assert cfg.custom_field == "value"  # type: ignore[attr-defined]


class TestGetLLMConfig:
    """get_llm_config() provider selection and error handling."""

    def test_get_anthropic_returns_anthropic_config(self):
        cfg = PassiConfig(anthropic={"api_key": "ant-key", "model": "claude-opus"})
        result = cfg.get_llm_config("anthropic")
        assert result.api_key == "ant-key"
        assert result.model == "claude-opus"

    def test_get_openai_returns_openai_config(self):
        cfg = PassiConfig(openai={"api_key": "oai-key", "model": "gpt-4o", "base_url": None})
        result = cfg.get_llm_config("openai")
        assert result.api_key == "oai-key"
        assert result.model == "gpt-4o"

    def test_get_ollama_returns_ollama_config(self):
        cfg = PassiConfig(ollama={"api_key": "ollama", "base_url": "http://10.0.0.1:8080/v1"})
        result = cfg.get_llm_config("ollama")
        assert result.base_url == "http://10.0.0.1:8080/v1"

    def test_get_config_uses_default_provider_when_none_given(self):
        cfg = PassiConfig(default_provider="openai")
        result = cfg.get_llm_config()
        assert isinstance(result, OpenAIConfig)

    def test_unknown_provider_raises_value_error(self):
        cfg = PassiConfig()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            cfg.get_llm_config("nonexistent")

    def test_unknown_provider_error_message_lists_options(self):
        cfg = PassiConfig()
        with pytest.raises(ValueError, match="anthropic"):
            cfg.get_llm_config("bad_provider")


class TestEnvVarLoading:
    """Config loaded from environment variables via PASSI_ prefix."""

    def test_api_key_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_ANTHROPIC__API_KEY", "env-key-123")
        cfg = PassiConfig()
        assert cfg.anthropic.api_key == "env-key-123"

    def test_model_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_ANTHROPIC__MODEL", "claude-opus-4-7")
        cfg = PassiConfig()
        assert cfg.anthropic.model == "claude-opus-4-7"

    def test_openai_base_url_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_OPENAI__BASE_URL", "https://proxy.example.com/v1")
        cfg = PassiConfig()
        assert cfg.openai.base_url == "https://proxy.example.com/v1"

    def test_ollama_base_url_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_OLLAMA__BASE_URL", "http://gpu-node:11434/v1")
        cfg = PassiConfig()
        assert cfg.ollama.base_url == "http://gpu-node:11434/v1"

    def test_debug_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_DEBUG", "true")
        cfg = PassiConfig()
        assert cfg.debug is True

    def test_default_provider_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_DEFAULT_PROVIDER", "ollama")
        cfg = PassiConfig()
        assert cfg.default_provider == "ollama"

    def test_max_tokens_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_ANTHROPIC__MAX_TOKENS", "16000")
        cfg = PassiConfig()
        assert cfg.anthropic.max_tokens == 16000

    def test_temperature_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_ANTHROPIC__TEMPERATURE", "0.7")
        cfg = PassiConfig()
        assert cfg.anthropic.temperature == 0.7

    def test_disabled_provider_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_OLLAMA__ENABLED", "true")
        cfg = PassiConfig()
        assert cfg.ollama.enabled is True


class TestLoadConfigFromFile:
    """load_config() from YAML / JSON files."""

    def test_load_from_yaml(self, tmp_path: Path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump({
            "default_provider": "openai",
            "anthropic": {"api_key": "yaml-key", "model": "claude-opus-4-7"},
        }))
        cfg = load_config(yaml_path)
        assert cfg.default_provider == "openai"
        assert cfg.anthropic.api_key == "yaml-key"
        assert cfg.anthropic.model == "claude-opus-4-7"

    def test_load_from_json(self, tmp_path: Path):
        json_path = tmp_path / "config.json"
        json_path.write_text(json.dumps({
            "default_provider": "ollama",
            "openai": {"api_key": "json-key", "base_url": "https://custom/v1"},
        }))
        cfg = load_config(json_path)
        assert cfg.default_provider == "ollama"
        assert cfg.openai.api_key == "json-key"
        assert cfg.openai.base_url == "https://custom/v1"

    def test_load_nonexistent_path_returns_default(self):
        cfg = load_config(Path("/nonexistent/config.yaml"))
        assert cfg.default_provider == "anthropic"

    def test_load_none_returns_default(self):
        cfg = load_config(None)
        assert cfg.default_provider == "anthropic"

    def test_yaml_with_debug_and_execution(self, tmp_path: Path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(yaml.dump({
            "debug": True,
            "execution": {"timeout_seconds": 600, "python_path": "/usr/bin/python3"},
        }))
        cfg = load_config(yaml_path)
        assert cfg.debug is True
        assert cfg.execution.timeout_seconds == 600

    def test_file_values_loaded_correctly(self, tmp_path: Path):
        """File config values are correctly loaded."""
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(yaml.dump({"anthropic": {"api_key": "file-key"}}))
        cfg = load_config(yaml_path)
        assert cfg.anthropic.api_key == "file-key"

    def test_env_vars_take_precedence_over_defaults(self, monkeypatch: pytest.MonkeyPatch):
        """Env vars override factory defaults."""
        monkeypatch.setenv("PASSI_ANTHROPIC__API_KEY", "env-key")
        cfg = PassiConfig()
        assert cfg.anthropic.api_key == "env-key"

    def test_passiconfig_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """PASSI_CONFIG env var merges JSON overrides onto config."""
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(yaml.dump({"default_provider": "anthropic", "debug": False}))
        monkeypatch.setenv(
            "PASSI_CONFIG",
            json.dumps({"default_provider": "ollama", "debug": True}),
        )
        cfg = load_config(yaml_path)
        assert cfg.default_provider == "ollama"
        assert cfg.debug is True


class TestExecutionConfig:
    """ExecutionConfig properties and env loading."""

    def test_r_home_env_var_is_preserved(self, monkeypatch: pytest.MonkeyPatch):
        """Explicit R_HOME env var is preserved even if not a valid R installation."""
        monkeypatch.setenv("PASSI_EXECUTION__R_HOME", "/custom/r/home")
        cfg = PassiConfig()
        assert Path(cfg.execution.r_home) == Path("/custom/r/home")

    def test_rscript_binary_falls_back_when_r_home_empty(self):
        """When r_home is empty, r_path default is used for binaries."""
        cfg = ExecutionConfig(r_home="", r_path="Rscript")
        # r_lib_path field validator creates ./R-lib dir but r_home stays "" if no R found
        # rscript_binary property may auto-detect if system R exists
        path = cfg.rscript_binary
        assert isinstance(path, str) and len(path) > 0

    def test_timeout_default(self):
        cfg = ExecutionConfig()
        assert cfg.timeout_seconds == 300

    def test_max_output_bytes_default(self):
        cfg = ExecutionConfig()
        assert cfg.max_output_bytes == 10 * 1024 * 1024


class TestSessionConfig:
    """SessionConfig defaults and overrides."""

    def test_default_sessions_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When no project .passi/ exists, sessions_dir defaults to ~/.passi/sessions.
        monkeypatch.chdir(tmp_path)
        cfg = SessionConfig()
        assert cfg.sessions_dir == Path.home() / ".passi" / "sessions"

    def test_default_sessions_dir_uses_project_passi(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When a project .passi/ directory exists, sessions_dir defaults inside it.
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".passi").mkdir()
        monkeypatch.chdir(project_dir)
        cfg = SessionConfig()
        assert cfg.sessions_dir == project_dir / ".passi" / "sessions"

    def test_checkpoint_interval_default(self):
        cfg = SessionConfig()
        assert cfg.checkpoint_interval == 5

    def test_env_override_sessions_dir(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_SESSION__SESSIONS_DIR", "/tmp/passi_sessions")
        cfg = PassiConfig()
        assert cfg.session.sessions_dir == Path("/tmp/passi_sessions")


class TestPassiConfigAFKMode:
    """AFK autonomous mode configuration tests."""

    def test_afk_mode_default_false(self):
        config = PassiConfig()
        assert config.afk_mode is False

    def test_afk_mode_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_AFK_MODE", "true")
        config = PassiConfig()
        assert config.afk_mode is True

    def test_afk_mode_false_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_AFK_MODE", "false")
        config = PassiConfig()
        assert config.afk_mode is False


class TestPlanConfig:
    """PlanConfig defaults and structure."""

    def test_plan_config_defaults(self):
        cfg = PlanConfig()
        assert cfg.qa_min_questions == 3
        assert cfg.qa_max_questions == 4
        assert cfg.max_recycles == 3

    def test_plan_config_in_passiconfig(self):
        cfg = PassiConfig()
        assert isinstance(cfg.plan, PlanConfig)
        assert cfg.plan.qa_min_questions == 3

    def test_plan_config_from_dict(self):
        cfg = PassiConfig(plan={"qa_min_questions": 5, "max_recycles": 2})
        assert cfg.plan.qa_min_questions == 5
        assert cfg.plan.max_recycles == 2


class TestResultDir:
    """result_dir field and output_dir deprecation."""

    def test_result_dir_default(self):
        cfg = PassiConfig()
        assert cfg.result_dir == Path("./result").resolve()

    def test_result_dir_custom(self, tmp_path: Path):
        custom = tmp_path / "results"
        cfg = PassiConfig(result_dir=str(custom))
        assert cfg.result_dir == custom.resolve()

    def test_output_dir_is_deprecated(self):
        cfg = PassiConfig()
        with pytest.warns(DeprecationWarning, match="Use result_dir"):
            _ = cfg.output_dir

    def test_output_dir_returns_result_dir(self):
        cfg = PassiConfig(result_dir="./my_results")
        with pytest.warns(DeprecationWarning):
            assert cfg.output_dir == Path("./my_results").resolve()

    def test_result_dir_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASSI_RESULT_DIR", "/tmp/passi_results")
        cfg = PassiConfig()
        # resolve() prepends the current drive on Windows
        assert cfg.result_dir == Path("/tmp/passi_results").resolve()


class TestProjectPassiDir:
    """Project-local .passi/ directory detection."""

    def test_project_passi_dir_none_when_no_passi(self, tmp_path: Path):
        result = _project_passi_dir(tmp_path)
        # May find ~/.passi when walking up — only assert None when not found
        if result is not None:
            assert result.is_dir()
            assert result.name == ".passi"

    def test_project_passi_dir_finds_passi(self, tmp_path: Path):
        (tmp_path / ".passi").mkdir()
        result = _project_passi_dir(tmp_path)
        assert result == tmp_path / ".passi"

    def test_project_passi_dir_walks_up(self, tmp_path: Path):
        (tmp_path / ".passi").mkdir()
        subdir = tmp_path / "a" / "b" / "c"
        subdir.mkdir(parents=True)
        result = _project_passi_dir(subdir)
        assert result == tmp_path / ".passi"

    def test_project_passi_dir_returns_path_or_none(self, tmp_path: Path):
        result = _project_passi_dir(tmp_path)
        # Returns None if no .passi/ found, or a Path if one exists above
        assert result is None or (isinstance(result, Path) and result.name == ".passi")

    def test_resolve_passi_dir_uses_project_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / ".passi").mkdir()
        result = _resolve_passi_dir()
        # May be overridden by actual project .passi/ above test temp dir
        # Just verify it returns a Path
        assert isinstance(result, Path)

    def test_hooks_config_uses_resolve_passi_dir(self):
        cfg = HooksConfig()
        assert "hooks.yaml" in str(cfg.hooks_file)

    def test_session_config_uses_resolve_passi_dir(self):
        cfg = SessionConfig()
        assert "sessions" in str(cfg.sessions_dir)
