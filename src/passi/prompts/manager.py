"""PromptManager — loads and composes prompt templates.

Uses string.Template for simple $var substitution. Templates are plain .txt files
so scientists can edit them without touching code. Supports hot-reload for debugging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from string import Template
from typing import ClassVar

logger = logging.getLogger(__name__)

# Map domain strings to template file names (without .txt extension)
_DOMAIN_TEMPLATE_MAP: dict[str, str] = {
    "transcriptomics": "domain_transcriptomics",
    "genomics": "domain_genomics",
    "epigenetics": "domain_epigenetics",
    "clinical": "domain_clinical",
    "multi-omics": "domain_transcriptomics",  # fallback to most common
    "proteomics": "domain_transcriptomics",
    "metabolomics": "domain_transcriptomics",
}


class PromptManager:
    """Load and compose system prompt from .txt template files.

    Templates use Python string.Template syntax ($var / ${var}).
    safe_substitute() is used so missing variables become literal $var — no crash.

    Usage:
        pm = PromptManager()
        prompt = pm.build_system_prompt(domain="transcriptomics")
        pm.reload()  # hot-reload after editing .txt files
    """

    # Templates that are always included
    _CORE_TEMPLATES: ClassVar[list[str]] = [
        "base_system",
        "tool_use_guidelines",
    ]

    # Templates included based on flags
    _OPTIONAL_TEMPLATES: ClassVar[dict[str, str]] = {
        "plan_enabled": "plan_mode",
        "data_check_enabled": "data_format_check",
    }

    def __init__(self, template_dir: str | Path | None = None) -> None:
        if template_dir:
            self._template_dir = Path(template_dir)
        else:
            # Default: prompts/ directory relative to this file
            self._template_dir = Path(__file__).resolve().parent
        self._cache: dict[str, Template] = {}
        self._load_all()

    # ── Public API ──

    def build_system_prompt(
        self,
        domain: str = "multi-omics",
        plan_enabled: bool = True,
        data_check_enabled: bool = True,
        **extra_vars: str,
    ) -> str:
        """Compose the full system prompt from templates.

        Args:
            domain: Analysis domain (transcriptomics, genomics, etc.)
            plan_enabled: Include plan-mode instructions
            data_check_enabled: Include data format check instructions
            **extra_vars: Additional template variables

        Returns:
            Composed system prompt string
        """
        variables: dict[str, str] = {"domain": domain, **extra_vars}
        parts: list[str] = []

        # Core templates (always included)
        for name in self._CORE_TEMPLATES:
            tmpl = self._cache.get(name)
            if tmpl is not None:
                parts.append(tmpl.safe_substitute(variables))

        # Optional templates (conditional)
        flags: dict[str, bool] = {
            "plan_enabled": plan_enabled,
            "data_check_enabled": data_check_enabled,
        }
        for flag_name, template_name in self._OPTIONAL_TEMPLATES.items():
            if flags.get(flag_name):
                tmpl = self._cache.get(template_name)
                if tmpl is not None:
                    parts.append(tmpl.safe_substitute(variables))

        # Domain template
        domain_template_name = _DOMAIN_TEMPLATE_MAP.get(domain, "domain_transcriptomics")
        domain_tmpl = self._cache.get(domain_template_name)
        if domain_tmpl is not None:
            parts.append(domain_tmpl.safe_substitute(variables))

        return "\n\n".join(parts)

    def reload(self) -> None:
        """Hot-reload: clear cache and re-read all template files from disk."""
        self._cache.clear()
        self._load_all()
        logger.info("Prompt templates reloaded from %s", self._template_dir)

    def get_raw(self, name: str) -> str | None:
        """Get raw template text for debugging. Name without .txt extension."""
        path = self._template_dir / f"{name}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def list_templates(self) -> list[str]:
        """List available template names (without .txt extension)."""
        return sorted(self._cache.keys())

    # ── Internal ──

    def _load_all(self) -> None:
        """Load all .txt files from the template directory into cache."""
        if not self._template_dir.exists():
            logger.warning("Template directory not found: %s", self._template_dir)
            return

        for path in self._template_dir.glob("*.txt"):
            name = path.stem  # filename without .txt
            try:
                text = path.read_text(encoding="utf-8")
                self._cache[name] = Template(text)
            except Exception as e:
                logger.warning("Failed to load template %s: %s", path.name, e)
