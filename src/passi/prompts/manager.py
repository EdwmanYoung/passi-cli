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
    "metabolomics": "domain_metabolomics",
}


class PromptManager:
    """Load and compose system prompt from .txt template files.

    Templates use Python string.Template syntax ($var / ${var}).
    safe_substitute() is used so missing variables become literal $var — no crash.

    Usage:
        pm = PromptManager()
        prompt = pm.build_system_prompt(domain="transcriptomics")
        pm.load_skill("metabolomics")
        pm.reload()  # hot-reload after editing .txt files
    """

    # Templates that are always included
    _CORE_TEMPLATES: ClassVar[list[str]] = [
        "base_system",
        "tool_use_guidelines",
        "bioinfo_principles",
    ]

    # Templates included based on flags
    _OPTIONAL_TEMPLATES: ClassVar[dict[str, str]] = {
        "plan_enabled": "plan_mode",
        "data_check_enabled": "data_format_check",
    }

    # Special templates for plan-first mode
    _PLAN_FIRST_TEMPLATE: ClassVar[str] = "plan_first"
    _PLAN_QA_TEMPLATE: ClassVar[str] = "plan_qa"
    _STEP_CONFIRM_TEMPLATE: ClassVar[str] = "step_confirm"

    _AVAILABLE_SKILLS: ClassVar[list[str]] = [
        "metabolomics",
        "pathway",
        "stats",
        "qc",
        "multi_omics",
    ]

    def __init__(self, template_dir: str | Path | None = None) -> None:
        if template_dir:
            self._template_dir = Path(template_dir)
        else:
            # Default: prompts/ directory relative to this file
            self._template_dir = Path(__file__).resolve().parent
        self._cache: dict[str, Template] = {}
        self._skill_cache: dict[str, Template] = {}
        self._active_skills: list[str] = []
        self._load_all()
        self._load_skills()

    # ── Public API ──

    def build_system_prompt(
        self,
        domain: str = "multi-omics",
        plan_enabled: bool = True,
        data_check_enabled: bool = True,
        afk_mode: bool = False,
        plan_first: bool = False,
        plan_qa: bool = False,
        step_confirm: bool = False,
        result_id: str = "",
        **extra_vars: str,
    ) -> str:
        """Compose the full system prompt from templates.

        Args:
            domain: Analysis domain (transcriptomics, genomics, etc.)
            plan_enabled: Include plan-mode instructions
            data_check_enabled: Include data format check instructions
            afk_mode: AFK autonomous mode (overrides plan/data_check templates)
            plan_first: Plan-first mode — agent must create plan before execution
            plan_qa: Pre-plan Q&A mode — agent must ask clarifying questions first
            step_confirm: Step confirmation mode — agent must confirm each step
            result_id: Current result directory ID (e.g., "result_20260626_141530")
            **extra_vars: Additional template variables

        Returns:
            Composed system prompt string
        """
        variables: dict[str, str] = {"domain": domain, "result_dir": result_id, **extra_vars}
        parts: list[str] = []

        # Core templates (always included)
        for name in self._CORE_TEMPLATES:
            tmpl = self._cache.get(name)
            if tmpl is not None:
                parts.append(tmpl.safe_substitute(variables))

        # Optional templates (conditional)
        if afk_mode:
            tmpl = self._cache.get("afk_mode")
            if tmpl is not None:
                parts.append(tmpl.safe_substitute(variables))
        else:
            flags: dict[str, bool] = {
                "plan_enabled": plan_enabled,
                "data_check_enabled": data_check_enabled,
            }
            for flag_name, template_name in self._OPTIONAL_TEMPLATES.items():
                if flags.get(flag_name):
                    tmpl = self._cache.get(template_name)
                    if tmpl is not None:
                        parts.append(tmpl.safe_substitute(variables))

        # Plan-first directive (if enabled)
        if plan_first and not afk_mode:
            tmpl = self._cache.get(self._PLAN_FIRST_TEMPLATE)
            if tmpl is not None:
                parts.append(tmpl.safe_substitute(variables))

        # Pre-plan Q&A directive (if plan QA mode is active)
        if plan_qa and plan_first and not afk_mode:
            tmpl = self._cache.get(self._PLAN_QA_TEMPLATE)
            if tmpl is not None:
                parts.append(tmpl.safe_substitute(variables))

        # Step confirmation protocol (if step confirm mode is active)
        if step_confirm and plan_first and not afk_mode:
            tmpl = self._cache.get(self._STEP_CONFIRM_TEMPLATE)
            if tmpl is not None:
                parts.append(tmpl.safe_substitute(variables))

        # Domain template
        domain_template_name = _DOMAIN_TEMPLATE_MAP.get(domain, "domain_transcriptomics")
        domain_tmpl = self._cache.get(domain_template_name)
        if domain_tmpl is not None:
            parts.append(domain_tmpl.safe_substitute(variables))

        # Active skills (appended at end for prominence)
        for skill_name in self._active_skills:
            skill_tmpl = self._skill_cache.get(skill_name)
            if skill_tmpl is not None:
                parts.append(
                    f"## Active Skill: {skill_name}\n\n"
                    + skill_tmpl.safe_substitute(variables)
                )

        return "\n\n".join(parts)

    def load_skill(self, name: str) -> bool:
        """Activate a skill by name. Returns True if loaded successfully.

        Skills stack — calling load_skill() multiple times adds more expertise.
        Use clear_skills() to reset.
        """
        name = name.strip().lower()
        if name not in self._AVAILABLE_SKILLS:
            logger.warning("Unknown skill: %s. Available: %s", name, self._AVAILABLE_SKILLS)
            return False
        if name in self._active_skills:
            return True  # already active
        if name not in self._skill_cache:
            logger.warning("Skill template not found: %s.txt", name)
            return False
        self._active_skills.append(name)
        logger.info("Skill activated: %s", name)
        return True

    def unload_skill(self, name: str) -> bool:
        """Deactivate a specific skill."""
        name = name.strip().lower()
        if name in self._active_skills:
            self._active_skills.remove(name)
            return True
        return False

    def clear_skills(self) -> None:
        """Deactivate all skills."""
        self._active_skills.clear()

    @property
    def active_skills(self) -> list[str]:
        """Return list of currently active skill names."""
        return list(self._active_skills)

    @classmethod
    def available_skills(cls) -> list[str]:
        """Return list of all available skill names."""
        return list(cls._AVAILABLE_SKILLS)

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

    def _load_skills(self) -> None:
        """Load all skill templates from prompts/skills/ directory."""
        skills_dir = self._template_dir / "skills"
        if not skills_dir.exists():
            logger.info("Skills directory not found: %s", skills_dir)
            return

        for path in skills_dir.glob("*.txt"):
            name = path.stem
            try:
                text = path.read_text(encoding="utf-8")
                self._skill_cache[name] = Template(text)
            except Exception as e:
                logger.warning("Failed to load skill %s: %s", path.name, e)
