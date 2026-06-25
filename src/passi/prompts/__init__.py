"""Prompt templating for PassiAgent.

Templates are plain .txt files in this directory using Python string.Template
syntax ($var). Edit them to tweak agent behavior without changing code.

Usage:
    from passi.prompts import PromptManager

    pm = PromptManager()
    prompt = pm.build_system_prompt(domain="transcriptomics")
    pm.reload()  # hot-reload after editing .txt files
"""

from passi.prompts.manager import PromptManager

__all__ = ["PromptManager"]
