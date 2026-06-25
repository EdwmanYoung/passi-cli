"""CLI entry point for PassiAgent.

Commands:
    passi chat              Start interactive chat (Rich TUI)
    passi ask <query>       Single query, stdout response
    passi server            Start web API server (reserved)
    passi session list      List sessions
    passi session load <id> Load a session
    passi session delete <id> Delete a session
    passi tool <name> ...    Direct tool invocation
    passi knowledge ...      Query the knowledge base
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
import click

from passi import __version__
from passi.config import PassiConfig, load_config


@click.group()
@click.version_option(version=__version__, prog_name="passi")
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), help="Path to config file (YAML/JSON)")
@click.option("--provider", "-p", default=None, help="LLM provider (anthropic, openai, ollama)")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
@click.pass_context
def main(ctx: click.Context, config_path: str | None, provider: str | None, debug: bool) -> None:
    """PassiAgent — Multi-omics bioinformatics analysis agent.

    Interactive AI-powered analysis for genomics, transcriptomics,
    epigenetics, proteomics, metabolomics, and clinical statistics.
    """
    ctx.ensure_object(dict)

    # Ensure ~/.passi/ exists before loading config
    from passi.config import _ensure_passi_home
    _ensure_passi_home()

    # Load configuration
    config = load_config(config_path)
    if debug:
        config = PassiConfig(**{**config.model_dump(), "debug": True})
    if provider:
        config = PassiConfig(**{**config.model_dump(), "default_provider": provider})

    ctx.obj["config"] = config


# ═══════════════════════════════════════════════════════════════
# chat — Interactive CLI mode
# ═══════════════════════════════════════════════════════════════

@main.command()
@click.option("--domain", "-d", default="multi-omics", help="Analysis domain")
@click.option("--mode", "-m", "start_mode", type=click.Choice(["chat", "plan", "afk"]), default=None, help="Agent mode on startup")
@click.option("--skills", "-s", default=None, help="Skills to activate on startup (comma-separated)")
@click.pass_context
def chat(ctx: click.Context, domain: str, start_mode: str | None, skills: str | None) -> None:
    """Start interactive chat mode (Rich TUI)."""
    from passi.ui.cli import PassiCLI

    config: PassiConfig = ctx.obj["config"]

    # Apply CLI options to config before creating CLI
    if start_mode == "afk":
        config = PassiConfig(**{**config.model_dump(), "afk_mode": True})

    cli = PassiCLI(config)

    # Apply mode and skills after CLI/agent initialization
    if start_mode or skills:
        # We need to wait for agent initialization, so we pass them as prefs
        cli._start_mode = start_mode
        cli._start_skills = [s.strip() for s in skills.split(",")] if skills else None

    try:
        asyncio.run(cli.start())
    except KeyboardInterrupt:
        click.echo("\nInterrupted.")


# ═══════════════════════════════════════════════════════════════
# ask — Non-interactive single query
# ═══════════════════════════════════════════════════════════════

@main.command()
@click.argument("query")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json", "markdown"]), default="text", help="Output format")
@click.option("--domain", "-d", default="multi-omics", help="Analysis domain")
@click.option("--afk", is_flag=True, help="AFK autonomous mode: auto-plan, auto-execute, never ask user")
@click.pass_context
def ask(ctx: click.Context, query: str, output_format: str, domain: str, afk: bool) -> None:
    """Send a single query and print the response (non-interactive)."""
    from passi.ui.print_mode import run_print_mode_sync

    config: PassiConfig = ctx.obj["config"]
    if afk:
        config = PassiConfig(**{**config.model_dump(), "afk_mode": True})
    exit_code = run_print_mode_sync(query, config, output_format, domain)
    sys.exit(exit_code)


# ═══════════════════════════════════════════════════════════════
# afk — AFK autonomous analysis
# ═══════════════════════════════════════════════════════════════

@main.command()
@click.argument("query")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json", "markdown"]), default="text", help="Output format")
@click.option("--domain", "-d", default="multi-omics", help="Analysis domain")
@click.pass_context
def afk(ctx: click.Context, query: str, output_format: str, domain: str) -> None:
    """Run analysis in AFK autonomous mode (auto-plan, auto-execute, no user prompts)."""
    from passi.ui.print_mode import run_print_mode_sync

    config: PassiConfig = ctx.obj["config"]
    config = PassiConfig(**{**config.model_dump(), "afk_mode": True})
    exit_code = run_print_mode_sync(query, config, output_format, domain)
    sys.exit(exit_code)


# ═══════════════════════════════════════════════════════════════
# server — Web API server (reserved)
# ═══════════════════════════════════════════════════════════════

@main.command()
@click.option("--host", default="127.0.0.1", help="Bind address")
@click.option("--port", default=8000, help="Bind port")
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload")
@click.pass_context
def server(ctx: click.Context, host: str, port: int, reload: bool) -> None:
    """Start the web API server (reserved for future use)."""
    try:
        from passi.api.server import create_app

        config: PassiConfig = ctx.obj["config"]
        app = create_app(config)
        import uvicorn

        uvicorn.run(app, host=host, port=port, reload=reload)
    except ImportError:
        click.echo("Web API dependencies not installed. Run: pip install digitagent[all]")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# session — Session management
# ═══════════════════════════════════════════════════════════════

@main.group()
def session() -> None:
    """Manage analysis sessions."""
    pass


@session.command("list")
@click.pass_context
def session_list(ctx: click.Context) -> None:
    """List all sessions."""
    from passi.infra.session import SessionManager

    config: PassiConfig = ctx.obj["config"]
    mgr = SessionManager(config)
    sessions = mgr.list_sessions()
    if not sessions:
        click.echo("No sessions found.")
        return
    click.echo(f"{'ID':<30} {'Domain':<18} {'Messages':<10} {'Created'}")
    click.echo("-" * 80)
    for s in sessions:
        click.echo(f"{s['session_id']:<30} {s['domain']:<18} {s['message_count']:<10} {s['created_at']}")


@session.command("load")
@click.argument("session_id")
@click.pass_context
def session_load(ctx: click.Context, session_id: str) -> None:
    """Load and restore a session."""
    from passi.infra.session import SessionManager

    config: PassiConfig = ctx.obj["config"]
    mgr = SessionManager(config)
    try:
        meta = mgr.load_session(session_id)
        click.echo(f"Session loaded: {meta.session_id} ({meta.domain}, {meta.message_count} messages)")
    except FileNotFoundError:
        click.echo(f"Session not found: {session_id}", err=True)
        sys.exit(1)


@session.command("delete")
@click.argument("session_id")
@click.option("--force", is_flag=True, help="Force deletion without confirmation")
@click.pass_context
def session_delete(ctx: click.Context, session_id: str, force: bool) -> None:
    """Delete a session and all its data."""
    if not force and not click.confirm(f"Delete session '{session_id}' and all its data?"):
        return
    from passi.infra.session import SessionManager

    config: PassiConfig = ctx.obj["config"]
    mgr = SessionManager(config)
    mgr.delete_session(session_id)
    click.echo(f"Session deleted: {session_id}")


# ═══════════════════════════════════════════════════════════════
# tool — Direct tool invocation
# ═══════════════════════════════════════════════════════════════

@main.group()
def tool() -> None:
    """Invoke tools directly."""
    pass


def _build_registry(config: PassiConfig) -> ToolRegistry:
    """Build a complete tool registry from config (shared by tool commands)."""
    from passi.tools.registry import ToolRegistry
    from passi.tools.io_tools import (
        ExportResultsTool,
        ParseOmicsDataTool,
        ReadFileTool,
        WriteFileTool,
    )
    from passi.tools.exec_tools import RunPythonTool, RunRTool
    from passi.tools.qc_tools import QcReportTool
    from passi.tools.genomics_tools import GwasAnalysisTool, ManhattanPlotTool, VcfStatsTool
    from passi.tools.epigenetics_tools import MethylationAnalysisTool, PeakQcTool
    from passi.tools.transcriptomics_tools import DifferentialAnalysisTool
    from passi.tools.enrichment_tools import EnrichmentTool
    from passi.tools.clinical_tools import SurvivalAnalysisTool

    exec_cfg = config.execution
    run_r = RunRTool()
    run_r.r_home = exec_cfg.r_home or ""
    run_r.r_lib_path = exec_cfg.r_lib_path or ""
    run_r.r_path = exec_cfg.rscript_binary

    de_tool = DifferentialAnalysisTool(
        r_home=exec_cfg.r_home or "",
        r_lib_path=exec_cfg.r_lib_path or "",
        r_path=exec_cfg.rscript_binary,
    )

    surv_tool = SurvivalAnalysisTool(
        r_home=exec_cfg.r_home or "",
        r_lib_path=exec_cfg.r_lib_path or "",
        r_path=exec_cfg.rscript_binary,
    )

    enrich_tool = EnrichmentTool(
        r_home=exec_cfg.r_home or "",
        r_lib_path=exec_cfg.r_lib_path or "",
        r_path=exec_cfg.rscript_binary,
    )

    registry = ToolRegistry()
    registry.register(ReadFileTool(), "io")
    registry.register(WriteFileTool(), "io")
    registry.register(ParseOmicsDataTool(), "io")
    registry.register(ExportResultsTool(), "io")
    registry.register(RunPythonTool(), "exec")
    registry.register(run_r, "exec")
    registry.register(QcReportTool(), "qc")
    registry.register(VcfStatsTool(), "genomics")
    registry.register(GwasAnalysisTool(), "genomics")
    registry.register(ManhattanPlotTool(), "genomics")
    registry.register(PeakQcTool(), "epigenetics")
    registry.register(MethylationAnalysisTool(), "epigenetics")
    registry.register(de_tool, "transcriptomics")
    registry.register(enrich_tool, "transcriptomics")
    registry.register(surv_tool, "clinical")
    return registry


@tool.command("list")
@click.option("--category", "-c", default=None, help="Filter by category")
@click.pass_context
def tool_list(ctx: click.Context, category: str | None) -> None:
    """List available tools."""
    config: PassiConfig = ctx.obj["config"]
    registry = _build_registry(config)

    tools = registry.list_tools(category=category)
    if not tools:
        click.echo(f"No tools found. Available categories: {list(registry.list_categories().keys())}")
        return
    click.echo("Available tools:")
    for t in tools:
        tool_instance = registry.get(t)
        if tool_instance:
            click.echo(f"  {tool_instance.name:<25} [{category or 'all'}] {tool_instance.description[:60]}")


@tool.command("run")
@click.argument("tool_name")
@click.argument("params_json", default="{}")
@click.pass_context
def tool_run(ctx: click.Context, tool_name: str, params_json: str) -> None:
    """Execute a tool directly with JSON parameters."""
    config: PassiConfig = ctx.obj["config"]
    registry = _build_registry(config)

    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        click.echo(f"Invalid JSON: {params_json}", err=True)
        sys.exit(1)

    result = registry.execute_sync(tool_name, params)
    click.echo(json.dumps(result, ensure_ascii=False, default=str, indent=2))


# ═══════════════════════════════════════════════════════════════
# knowledge — Knowledge base queries
# ═══════════════════════════════════════════════════════════════

@main.group()
def knowledge() -> None:
    """Query the knowledge base."""
    pass


@knowledge.command("search")
@click.argument("query")
@click.pass_context
def knowledge_search(ctx: click.Context, query: str) -> None:
    """Search methods by keyword."""
    from passi.knowledge.methods import search_methods

    results = search_methods(query)
    if not results:
        click.echo(f"No methods found for: {query}")
        return
    click.echo(f"Methods matching '{query}':")
    for mid, info in results.items():
        click.echo(f"  {mid:<25} {info['name']:<20} [{info['backend']}] {info.get('domain', '')}")


@knowledge.command("methods")
@click.option("--domain", "-d", default=None, help="Filter by omics domain")
@click.pass_context
def knowledge_methods(ctx: click.Context, domain: str | None) -> None:
    """List analysis methods, optionally filtered by domain."""
    from passi.knowledge.methods import get_methods_by_domain, list_all_methods

    if domain:
        methods = get_methods_by_domain(domain)
        click.echo(f"Methods for domain '{domain}':")
    else:
        catalog = list_all_methods()
        click.echo("Methods by category:")
        for cat, method_ids in catalog.items():
            click.echo(f"\n  {cat}: {len(method_ids)} methods")
            for mid in method_ids[:10]:
                click.echo(f"    - {mid}")
            if len(method_ids) > 10:
                click.echo(f"    ... and {len(method_ids) - 10} more")
        return

    for mid, info in methods.items():
        click.echo(f"  {mid:<25} {info['name']:<20} [{info['backend']}]")


@knowledge.command("formats")
@click.option("--domain", "-d", default=None, help="Filter by omics domain")
@click.pass_context
def knowledge_formats(ctx: click.Context, domain: str | None) -> None:
    """List supported data formats, optionally filtered by domain."""
    from passi.knowledge.formats import get_formats_by_domain, list_all_formats

    if domain:
        formats = get_formats_by_domain(domain)
        click.echo(f"Formats for domain '{domain}':")
        for f in formats:
            click.echo(f"  {f['format']:<20} {', '.join(f['suffixes']):<30} {f['description']}")
    else:
        catalog = list_all_formats()
        click.echo("Formats by domain:")
        for dom, formats in catalog.items():
            click.echo(f"\n  {dom}:")
            for f in formats[:8]:
                click.echo(f"    {f['format']:<20} {', '.join(f['suffixes'])}")
            if len(formats) > 8:
                click.echo(f"    ... and {len(formats) - 8} more")


# ═══════════════════════════════════════════════════════════════
# run — Execute a pipeline workflow
# ═══════════════════════════════════════════════════════════════

@main.command()
@click.argument("pipeline_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="./output", help="Output directory")
@click.pass_context
def run(ctx: click.Context, pipeline_path: str, output: str) -> None:
    """Execute a predefined analysis pipeline (YAML workflow)."""
    import yaml

    from passi.knowledge.pipelines import get_pipeline

    # Check if it's a named pipeline
    path = Path(pipeline_path)
    if path.suffix in (".yaml", ".yml"):
        with open(path, encoding="utf-8") as f:
            workflow = yaml.safe_load(f)
        pipeline_name = path.stem
    else:
        workflow = get_pipeline(pipeline_path)
        pipeline_name = pipeline_path

    if workflow is None:
        click.echo(f"Pipeline not found: {pipeline_path}", err=True)
        click.echo("Available pipelines:")
        from passi.knowledge.pipelines import list_pipelines
        for p in list_pipelines():
            click.echo(f"  {p['name']}: {p['title']}")
        sys.exit(1)

    click.echo(f"Running pipeline: {workflow.get('name', pipeline_name)}")
    click.echo(f"  Domain: {workflow.get('domain', 'unknown')}")
    click.echo(f"  Steps: {len(workflow.get('steps', []))}")
    click.echo(f"  Output: {output}")
    click.echo("\nPipeline execution will be available in Phase 2.")
    click.echo("For now, use interactive mode: passi chat")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
