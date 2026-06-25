"""Non-interactive (print) mode for PassiAgent.

Executes a single query and outputs the result to stdout.
Suitable for scripting, piping, and CI/CD integration.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from passi.config import PassiConfig
from passi.infra.runtime import Runtime
from passi.soul.passi_agent import PassiAgent


def _safe_print(text: str) -> None:
    """Print text, replacing characters unsupported by the console encoding."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


async def run_print_mode(
    query: str,
    config: PassiConfig | None = None,
    output_format: str = "text",
    domain: str = "multi-omics",
) -> int:
    """Execute a single query and print the result.

    Args:
        query: The analysis request
        config: PassiConfig (uses defaults if None)
        output_format: 'text', 'json', or 'markdown'
        domain: Analysis domain

    Returns:
        Exit code (0 = success)
    """
    if config is None:
        config = PassiConfig()

    runtime = Runtime(config)
    runtime.session.create_session(domain=domain)
    agent = PassiAgent(runtime)
    await agent.initialize()

    try:
        response = await agent.chat(query)

        if output_format == "json":
            _print_json(response)
        elif output_format == "markdown":
            _print_markdown(response)
        else:
            _print_text(response)

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        await agent.shutdown()


def _print_text(response: Any) -> None:
    """Print response as plain text."""
    # Check for pending question first
    pq = response.metadata.get("pending_question") if response.metadata else None
    if pq:
        _safe_print("\n[Agent needs your input]")
        _safe_print(f"Q: {pq['question']}")
        if pq.get("context"):
            _safe_print(f"Context: {pq['context']}")
        if pq.get("options"):
            _safe_print("Options:")
            for i, opt in enumerate(pq["options"], 1):
                _safe_print(f"  {i}. {opt}")
        return

    content = response.content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    _safe_print(block["text"])
                elif block.get("type") == "tool_use":
                    _safe_print(f"\n[Tool: {block.get('name', '?')}]")
    elif isinstance(content, str):
        _safe_print(content)


def _print_json(response: Any) -> None:
    """Print response as JSON."""
    output = {
        "role": response.role,
        "content": response.content,
        "metadata": response.metadata,
    }
    print(json.dumps(output, ensure_ascii=False, default=str))


def _print_markdown(response: Any) -> None:
    """Print response as markdown."""
    content = response.content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block["text"]
                    if text and not text.startswith(("#", ">", "-", "|")):
                        _safe_print(text)
                    else:
                        _safe_print(text)
                    _safe_print("")
    elif isinstance(content, str):
        _safe_print(content)


def run_print_mode_sync(
    query: str,
    config: PassiConfig | None = None,
    output_format: str = "text",
    domain: str = "multi-omics",
) -> int:
    """Synchronous wrapper for run_print_mode."""
    return asyncio.run(run_print_mode(query, config, output_format, domain))
