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
    content = response.content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    print(block["text"])
                elif block.get("type") == "tool_use":
                    print(f"\n[Tool: {block.get('name', '?')}]")
    elif isinstance(content, str):
        print(content)


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
                    # Ensure proper markdown formatting
                    if text and not text.startswith(("#", ">", "-", "|")):
                        print(text)
                    else:
                        print(text)
                    print()
    elif isinstance(content, str):
        print(content)


def run_print_mode_sync(
    query: str,
    config: PassiConfig | None = None,
    output_format: str = "text",
    domain: str = "multi-omics",
) -> int:
    """Synchronous wrapper for run_print_mode."""
    return asyncio.run(run_print_mode(query, config, output_format, domain))
