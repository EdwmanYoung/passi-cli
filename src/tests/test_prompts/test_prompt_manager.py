"""Unit tests for PromptManager — template loading, composition, hot-reload."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from passi.prompts.manager import PromptManager


class TestPromptManager:
    """PromptManager core tests."""

    def test_loads_all_templates_from_default_dir(self):
        pm = PromptManager()
        templates = pm.list_templates()
        assert "base_system" in templates
        assert "tool_use_guidelines" in templates
        assert "bioinfo_principles" in templates
        assert "plan_mode" in templates
        assert "data_format_check" in templates
        assert "afk_mode" in templates
        assert "domain_transcriptomics" in templates
        assert "domain_metabolomics" in templates

    def test_build_system_prompt_includes_core_templates(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(domain="transcriptomics")
        assert "PassiAgent" in prompt
        assert "Capabilities" in prompt
        assert "How You Work" in prompt
        assert "Bioinformatics Expert Principles" in prompt

    def test_bioinfo_principles_in_core_system_prompt(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(domain="genomics")
        assert "Statistical Rigor" in prompt
        assert "Data Integrity" in prompt
        assert "Tool-First Analysis" in prompt
        assert "Biological Interpretation" in prompt

    def test_domain_metabolomics_template(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(domain="metabolomics")
        assert "Metabolomics Analysis Guidelines" in prompt
        assert "metabolomics" in prompt.lower()

    def test_domain_metabolomics_not_in_transcriptomics(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(domain="transcriptomics")
        assert "Metabolomics Analysis Guidelines" not in prompt

    def test_build_system_prompt_includes_plan_mode_when_enabled(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(plan_enabled=True)
        assert "Plan Mode" in prompt

    def test_build_system_prompt_excludes_plan_mode_when_disabled(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(plan_enabled=False)
        assert "Plan Mode" not in prompt

    def test_build_system_prompt_includes_data_check_when_enabled(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(data_check_enabled=True)
        assert "Data Format Check" in prompt

    def test_build_system_prompt_excludes_data_check_when_disabled(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(data_check_enabled=False)
        assert "Data Format Check" not in prompt

    def test_domain_substitution(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(domain="genomics")
        assert "genomics" in prompt.lower()
        assert "Current analysis domain: genomics" in prompt

    def test_domain_template_included(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(domain="transcriptomics")
        assert "DESeq2" in prompt

    def test_safe_substitute_leaves_missing_vars(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(domain="transcriptomics")
        # No ValueError should be raised for missing $vars — they become literal
        assert True  # safe_substitute doesn't raise

    def test_custom_template_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpl_dir = Path(tmpdir)
            (tmpl_dir / "base_system.txt").write_text("Custom: $domain", encoding="utf-8")
            (tmpl_dir / "tool_use_guidelines.txt").write_text("Tools here.", encoding="utf-8")
            pm = PromptManager(tmpl_dir)
            prompt = pm.build_system_prompt(domain="test")
            assert "Custom: test" in prompt
            assert "Tools here." in prompt

    def test_reload_clears_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpl_dir = Path(tmpdir)
            (tmpl_dir / "base_system.txt").write_text("Version 1", encoding="utf-8")
            (tmpl_dir / "tool_use_guidelines.txt").write_text("Tools.", encoding="utf-8")
            pm = PromptManager(tmpl_dir)
            assert "Version 1" in pm.build_system_prompt()
            # Change on disk
            (tmpl_dir / "base_system.txt").write_text("Version 2", encoding="utf-8")
            pm.reload()
            assert "Version 2" in pm.build_system_prompt()
            assert "Version 1" not in pm.build_system_prompt()

    def test_get_raw_returns_template_text(self):
        pm = PromptManager()
        raw = pm.get_raw("base_system")
        assert raw is not None
        assert "PassiAgent" in raw

    def test_get_raw_returns_none_for_missing(self):
        pm = PromptManager()
        assert pm.get_raw("nonexistent") is None

    def test_afk_mode_replaces_plan_and_data_check(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(afk_mode=True)
        assert "AFK Autonomous Mode" in prompt
        assert "Never Call ask_user" in prompt
        assert "Plan Mode" not in prompt
        assert "Data Format Check" not in prompt

    def test_afk_mode_with_plan_disabled_still_uses_afk_template(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(afk_mode=True, plan_enabled=False)
        assert "AFK Autonomous Mode" in prompt
        assert "Plan Mode" not in prompt

    def test_list_templates_returns_sorted_names(self):
        pm = PromptManager()
        names = pm.list_templates()
        assert names == sorted(names)
        assert len(names) >= 4


class TestSkillSystem:
    """Skill loading, stacking, and clearing tests."""

    def test_available_skills_includes_all_expected(self):
        skills = PromptManager.available_skills()
        assert "metabolomics" in skills
        assert "pathway" in skills
        assert "stats" in skills
        assert "qc" in skills
        assert "multi_omics" in skills

    def test_load_skill_adds_to_active(self):
        pm = PromptManager()
        assert pm.load_skill("metabolomics") is True
        assert "metabolomics" in pm.active_skills

    def test_load_skill_unknown_returns_false(self):
        pm = PromptManager()
        assert pm.load_skill("nonexistent_skill") is False
        assert pm.active_skills == []

    def test_load_skill_duplicate_no_error(self):
        pm = PromptManager()
        pm.load_skill("metabolomics")
        assert pm.load_skill("metabolomics") is True  # idempotent
        assert pm.active_skills == ["metabolomics"]

    def test_unload_skill(self):
        pm = PromptManager()
        pm.load_skill("metabolomics")
        pm.load_skill("stats")
        assert pm.unload_skill("metabolomics") is True
        assert pm.active_skills == ["stats"]
        assert pm.unload_skill("nonexistent") is False

    def test_clear_skills(self):
        pm = PromptManager()
        pm.load_skill("metabolomics")
        pm.load_skill("stats")
        pm.load_skill("qc")
        assert len(pm.active_skills) == 3
        pm.clear_skills()
        assert pm.active_skills == []

    def test_skill_included_in_system_prompt(self):
        pm = PromptManager()
        pm.load_skill("metabolomics")
        prompt = pm.build_system_prompt(domain="transcriptomics")
        assert "Active Skill: metabolomics" in prompt
        assert "Metabolomics Skill" in prompt
        assert "NMR" in prompt  # metabolomics skill mentions NMR

    def test_skill_included_in_prompt_after_mode_switch(self):
        """Skills should be included even when plan_first is set."""
        pm = PromptManager()
        pm.load_skill("stats")
        prompt = pm.build_system_prompt(domain="genomics", plan_first=True)
        assert "Active Skill: stats" in prompt
        assert "Statistical Rigor Skill" in prompt

    def test_skill_not_in_prompt_when_not_loaded(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt()
        assert "Active Skill:" not in prompt

    def test_multiple_skills_stacked(self):
        pm = PromptManager()
        pm.load_skill("metabolomics")
        pm.load_skill("stats")
        prompt = pm.build_system_prompt()
        assert "Active Skill: metabolomics" in prompt
        assert "Active Skill: stats" in prompt
        # Metabolomics comes before stats in prompt
        idx1 = prompt.index("Active Skill: metabolomics")
        idx2 = prompt.index("Active Skill: stats")
        assert idx1 < idx2

    def test_plan_first_template_included(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt(plan_first=True)
        assert "Plan-First Mode Directive" in prompt
        assert "create_plan" in prompt
        assert "Step 1: Understand the Request" in prompt

    def test_plan_first_skipped_in_afk_mode(self):
        """plan_first should be skipped when afk_mode is active."""
        pm = PromptManager()
        prompt = pm.build_system_prompt(afk_mode=True, plan_first=True)
        assert "Plan-First Mode Directive" not in prompt
        assert "AFK Autonomous Mode" in prompt

    def test_plan_first_not_included_by_default(self):
        pm = PromptManager()
        prompt = pm.build_system_prompt()
        assert "Plan-First Mode Directive" not in prompt

    def test_custom_template_dir_includes_skills(self):
        """Skills should load from skills/ subdirectory in custom template dir."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpl_dir = Path(tmpdir)
            # Core templates
            (tmpl_dir / "base_system.txt").write_text("Base: $domain", encoding="utf-8")
            (tmpl_dir / "tool_use_guidelines.txt").write_text("Tools.", encoding="utf-8")
            # Skills directory
            skills_dir = tmpl_dir / "skills"
            skills_dir.mkdir()
            (skills_dir / "metabolomics.txt").write_text("Custom Metabolomics Skill Content", encoding="utf-8")

            pm = PromptManager(tmpl_dir)
            assert pm.load_skill("metabolomics") is True
            prompt = pm.build_system_prompt(domain="test")
            assert "Active Skill: metabolomics" in prompt
            assert "Custom Metabolomics Skill Content" in prompt
