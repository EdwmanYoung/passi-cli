"""E2E tests for metabolomics analysis — proj01 to proj05.

Each test exercises the full agent pipeline: plan creation, data reading,
format detection, code execution (Python/R), log inspection, and audit trail
preservation. Code execution run directories (scripts, logs, output files,
metadata) are verified for every test.

Run all:
    python -m pytest src/tests/test_integration/test_metabolomics_e2e.py -v -m "integration and slow" --timeout 600

Run single:
    python -m pytest src/tests/test_integration/test_metabolomics_e2e.py::TestMetabolomicsE2E::test_proj01_cachexia_case_control -v -m integration
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import pytest

from passi.config import PassiConfig
from passi.infra.runtime import Runtime
from passi.soul.passi_agent import PassiAgent

# ── Paths ──────────────────────────────────────────────────────────────

DATA_BASE = Path(__file__).resolve().parents[3] / "test_dataset" / "data" / "05_metabolomics"
RESULTS_BASE = Path(__file__).resolve().parents[3] / "e2e_results" / "metabolomics"

# ── Helpers ────────────────────────────────────────────────────────────


def _is_api_configured(provider: str = "anthropic") -> bool:
    cfg = PassiConfig()
    key = getattr(cfg, provider).api_key
    return bool(key) and "your-" not in key and "sk-your" not in key


def _skip_if_not_configured(provider: str = "anthropic") -> None:
    if not _is_api_configured(provider):
        pytest.skip(f"No valid {provider} API key in .env")


def _make_runtime(project_name: str) -> Runtime:
    """Create a Runtime with isolated session/output directories."""
    sessions_dir = RESULTS_BASE / project_name / "sessions"
    output_dir = RESULTS_BASE / project_name / "output"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = PassiConfig(
        anthropic={"api_key": PassiConfig().anthropic.api_key, "model": PassiConfig().anthropic.model},
        default_provider="anthropic",
        session={"sessions_dir": sessions_dir},
        output_dir=output_dir,
        debug=False,
    )
    return Runtime(config=cfg)


def _save_audit_trail(agent: PassiAgent, project_name: str, response_text: str) -> None:
    """Persist the complete audit trail and agent response."""
    out = RESULTS_BASE / project_name
    out.mkdir(parents=True, exist_ok=True)

    # Agent response as markdown
    (out / "agent_response.md").write_text(response_text, encoding="utf-8")

    # Copy wire.jsonl from the Wire's actual file path (CWD-relative by default)
    wire_path = agent.wire._wire_path
    if wire_path.exists():
        shutil.copy2(wire_path, out / "wire.jsonl")

    # Copy provenance.jsonl from the output directory
    output_dir = Path(agent.runtime.config.output_dir)
    prov_path = output_dir / "provenance.jsonl"
    if prov_path.exists():
        shutil.copy2(prov_path, out / "provenance.jsonl")

    # Copy session audit files (tasks.jsonl, plan.yaml, session.yaml)
    session = agent.runtime.session.active_session
    if session is not None:
        session_dir = agent.runtime.session.get_session_dir()
        for fname in ["tasks.jsonl", "plan.yaml", "session.yaml"]:
            src = session_dir / fname
            if src.exists():
                shutil.copy2(src, out / fname)

    # Write test metadata
    (out / "test_metadata.yaml").write_text(
        f"project: {project_name}\n"
        f"timestamp: {datetime.now().isoformat()}\n"
        "agent: PassiAgent\n",
        encoding="utf-8",
    )


def _extract_text(response) -> str:
    """Extract plain text from AgentMessage content."""
    parts: list[str] = []
    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    parts.append(f"\n[Tool: {block.get('name', '?')}]")
    elif isinstance(response.content, str):
        parts.append(response.content)
    return "\n".join(parts)


def _find_run_dirs(output_dir: Path) -> list[Path]:
    """Find all run directories created by code execution tools."""
    runs_base = output_dir / "runs"
    if not runs_base.exists():
        return []
    run_dirs: list[Path] = []
    for session_dir in runs_base.iterdir():
        if session_dir.is_dir():
            for run_dir in session_dir.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith("run_"):
                    run_dirs.append(run_dir)
    return sorted(run_dirs)


def _verify_run_dir(run_dir: Path, expected_script_ext: str) -> dict:
    """Verify a run directory has all expected files. Returns dict of findings."""
    findings: dict[str, bool | str] = {}
    script_file = run_dir / f"script.{expected_script_ext}"
    findings["script_exists"] = script_file.exists()
    findings["stdout_exists"] = (run_dir / "stdout.log").exists()
    findings["stderr_exists"] = (run_dir / "stderr.log").exists()
    findings["metadata_exists"] = (run_dir / "run_metadata.json").exists()

    if findings["metadata_exists"]:
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        findings["has_tool"] = "tool" in metadata
        findings["has_exit_code"] = "exit_code" in metadata
        findings["has_duration_ms"] = "duration_ms" in metadata
        findings["has_output_files"] = "output_files" in metadata
    return findings


def _count_code_exec_events(wire_path: Path) -> dict[str, int]:
    """Count run_python and run_r TOOL_CALL events in wire.jsonl."""
    counts: dict[str, int] = {"run_python": 0, "run_r": 0}
    if not wire_path.exists():
        return counts
    with open(wire_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "tool_call":
                name = event.get("data", {}).get("name", "")
                if name in counts:
                    counts[name] += 1
    return counts


# ═══════════════════════════════════════════════════════════════════════
# E2E Tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.slow
class TestMetabolomicsE2E:
    """E2E metabolomics analysis across 5 project types (4 analysis + 1 error recovery)."""

    # ── proj01: Case-Control ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_proj01_cachexia_case_control(self):
        """Human cachexia urine NMR: 63 metabolites, cachexic vs control."""
        _skip_if_not_configured("anthropic")

        data_file = DATA_BASE / "proj01_human_cachexia_urine_nmr" / "metaboanalyst_human_cachexia.csv"
        assert data_file.exists(), f"Data file not found: {data_file}"

        runtime = _make_runtime("proj01_human_cachexia")
        runtime.session.create_session(domain="metabolomics")

        query = (
            f"Analyze {data_file}. "
            "This is human urine 1H NMR metabolomics data with 63 metabolites and 77 samples. "
            "The first column is 'Patient ID' (sample identifier). "
            "The second column 'Muscle loss' is the group label: 'cachexic' (cancer cachexia, n=47) vs 'control' (n=30). "
            "Remaining columns are metabolite concentrations. "
            "Please follow these steps in order: "
            "(1) Create an analysis plan named 'Cachexia Metabolomics' using create_plan tool (title field required). "
            "(2) Use read_file to read first 15 lines. "
            "(3) Use parse_omics_data to detect format. "
            "(4) Use update_plan_status to mark steps done. "
            "(5) Use the run_python tool to write and execute Python code that reads the CSV with pandas, "
            "computes mean and std for each metabolite grouped by Muscle loss, and prints the top 5 "
            "metabolites with largest mean difference between groups. Print the results clearly. "
            "(6) After the code runs, use read_file to check the stdout.log in the run directory "
            "(the path is in the run_python result as run_dir). "
            "(7) Based on the code output, summarize: dimensions, group sizes, top differential metabolites."
        )
        print(f"\n  Query: {query[:200]}...")

        t0 = time.perf_counter()
        agent = PassiAgent(runtime)
        await agent.initialize()
        response = await agent.chat(query)
        elapsed = time.perf_counter() - t0

        text = _extract_text(response)
        print(f"\n  proj01 latency: {elapsed:.0f}s")
        print(f"  Response length: {len(text)} chars")

        _save_audit_trail(agent, "proj01_human_cachexia", text)
        await agent.shutdown()

        assert response.role == "agent"
        assert len(text) > 100, "Response too short — agent may have failed"
        # Verify audit files
        out = RESULTS_BASE / "proj01_human_cachexia"
        assert (out / "agent_response.md").exists()
        assert (out / "wire.jsonl").exists()
        # Verify code execution
        code_events = _count_code_exec_events(out / "wire.jsonl")
        assert code_events["run_python"] + code_events["run_r"] >= 1, \
            f"No code execution tool calls found in wire: {code_events}"
        run_dirs = _find_run_dirs(Path(agent.runtime.config.output_dir))
        assert len(run_dirs) >= 1, f"No run directories found under {agent.runtime.config.output_dir}"
        for rd in run_dirs:
            findings = _verify_run_dir(rd, expected_script_ext="py")
            assert findings["script_exists"], f"script.py missing in {rd}"
            assert findings["stdout_exists"], f"stdout.log missing in {rd}"
            assert findings["metadata_exists"], f"run_metadata.json missing in {rd}"

    # ── proj02: NMR Bins + Multi-file ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_proj02_kidney_disease_nmr_bins(self):
        """Human kidney disease urine NMR: 200 bins, patient vs control, + peak ZIP."""
        _skip_if_not_configured("anthropic")

        proj_dir = DATA_BASE / "proj02_human_kidney_disease_urine_nmr"
        bin_file = proj_dir / "metaboanalyst_nmr_bins.csv"
        peak_zip = proj_dir / "metaboanalyst_nmr_peaks.zip"
        assert bin_file.exists(), f"Bin file not found: {bin_file}"
        assert peak_zip.exists(), f"Peak ZIP not found: {peak_zip}"

        runtime = _make_runtime("proj02_human_kidney")
        runtime.session.create_session(domain="metabolomics")
        agent = PassiAgent(runtime)
        await agent.initialize()

        query = (
            f"Analyze the data in {proj_dir}. "
            "This directory contains human urine 1H NMR metabolomics data for kidney disease: "
            f"(a) {bin_file.name} — NMR spectral bin matrix (200 bins, 0.22-9.98 ppm, 50 samples), "
            f"(b) {peak_zip.name} — 50 peak list CSV files (grouped as Healthy vs Kidney_disease). "
            "Group: patient (n=25) vs control (n=25), severe kidney disease. "
            "Please follow these steps in order: "
            "(1) Create an analysis plan named 'Kidney Disease NMR Analysis' using create_plan tool (title field required). "
            "(2) Use read_file to read first 15 lines of the bin CSV. "
            "(3) Use parse_omics_data on the bin CSV. "
            "(4) Use update_plan_status to mark steps done. "
            "(5) Use the run_python tool to write and execute Python code that loads the bin CSV with pandas, "
            "computes the mean intensity per bin across all samples, and prints the 5 bins with "
            "highest mean intensity and their ppm values. Print the results clearly. "
            "(6) After the code runs, use read_file to check stdout.log in the run directory. "
            "(7) Based on the code output, summarize: dimensions, ppm range, group info, key spectral regions."
        )
        print(f"\n  Query: {query[:200]}...")

        t0 = time.perf_counter()
        response = await agent.chat(query)
        elapsed = time.perf_counter() - t0

        text = _extract_text(response)
        print(f"\n  proj02 latency: {elapsed:.0f}s")
        print(f"  Response length: {len(text)} chars")

        _save_audit_trail(agent, "proj02_human_kidney", text)
        await agent.shutdown()

        assert response.role == "agent"
        assert len(text) > 100
        out = RESULTS_BASE / "proj02_human_kidney"
        assert (out / "agent_response.md").exists()
        assert (out / "wire.jsonl").exists()
        # Verify code execution
        code_events = _count_code_exec_events(out / "wire.jsonl")
        assert code_events["run_python"] + code_events["run_r"] >= 1, \
            f"No code execution tool calls found: {code_events}"
        run_dirs = _find_run_dirs(Path(agent.runtime.config.output_dir))
        assert len(run_dirs) >= 1, f"No run directories found"
        for rd in run_dirs:
            findings = _verify_run_dir(rd, expected_script_ext="py")
            assert findings["script_exists"], f"script.py missing in {rd}"
            assert findings["stdout_exists"], f"stdout.log missing in {rd}"

    # ── proj03: Multi-Group ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_proj03_bovine_diet_multi_group(self):
        """Bovine diet rumen NMR: 47 metabolites, 4 diet groups (0/15/30/45% grain)."""
        _skip_if_not_configured("anthropic")

        data_file = DATA_BASE / "proj03_bovine_diet_rumen_nmr" / "metaboanalyst_cow_diet.csv"
        assert data_file.exists(), f"Data file not found: {data_file}"

        runtime = _make_runtime("proj03_bovine_diet")
        runtime.session.create_session(domain="metabolomics")
        agent = PassiAgent(runtime)
        await agent.initialize()

        query = (
            f"Analyze {data_file}. "
            "This is bovine rumen fluid 1H NMR metabolomics data with 47 metabolites and 39 samples. "
            "The first column is 'Sample' (sample ID). "
            "The second column 'Diet' is the group: 0, 15, 30, or 45 (percentage of grain in diet, 4 groups). "
            "Remaining columns are metabolite concentrations. "
            "Please follow these steps in order: "
            "(1) Create an analysis plan named 'Bovine Diet Metabolomics' using create_plan tool (title field required). "
            "(2) Use read_file to read first 15 lines. "
            "(3) Use parse_omics_data to detect format. "
            "(4) Use update_plan_status to mark steps done. "
            "(5) Use the run_python tool to write and execute Python code that reads the CSV with pandas, "
            "groups by Diet, computes mean of each metabolite per diet group, and prints the top 3 "
            "metabolites with largest variance across diet groups (suggesting dose-response). "
            "Print the results clearly. "
            "(6) After the code runs, use read_file to check stdout.log in the run directory. "
            "(7) Based on the code output, summarize: dimensions, group sizes per diet level, "
            "metabolites showing dose-response pattern with grain %."
        )
        print(f"\n  Query: {query[:200]}...")

        t0 = time.perf_counter()
        response = await agent.chat(query)
        elapsed = time.perf_counter() - t0

        text = _extract_text(response)
        print(f"\n  proj03 latency: {elapsed:.0f}s")
        print(f"  Response length: {len(text)} chars")

        _save_audit_trail(agent, "proj03_bovine_diet", text)
        await agent.shutdown()

        assert response.role == "agent"
        assert len(text) > 100
        out = RESULTS_BASE / "proj03_bovine_diet"
        assert (out / "agent_response.md").exists()
        assert (out / "wire.jsonl").exists()
        # Verify code execution
        code_events = _count_code_exec_events(out / "wire.jsonl")
        assert code_events["run_python"] + code_events["run_r"] >= 1, \
            f"No code execution tool calls found: {code_events}"
        run_dirs = _find_run_dirs(Path(agent.runtime.config.output_dir))
        assert len(run_dirs) >= 1, f"No run directories found"
        for rd in run_dirs:
            findings = _verify_run_dir(rd, expected_script_ext="py")
            assert findings["script_exists"], f"script.py missing in {rd}"
            assert findings["stdout_exists"], f"stdout.log missing in {rd}"

    # ── proj04: Paired Time Series ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_proj04_bovine_urine_paired_timeseries(self):
        """Bovine urine time-series NMR: 32 compounds, 7 cows x 2 time points, paired."""
        _skip_if_not_configured("anthropic")

        data_file = DATA_BASE / "proj04_bovine_urine_timeseries_nmr" / "metaboanalyst_time_series.csv"
        assert data_file.exists(), f"Data file not found: {data_file}"

        runtime = _make_runtime("proj04_bovine_timeseries")
        runtime.session.create_session(domain="metabolomics")
        agent = PassiAgent(runtime)
        await agent.initialize()

        query = (
            f"Analyze {data_file}. "
            "This is bovine urine 1H NMR time-series metabolomics data. "
            "Format: first column is compound name, remaining 14 columns are samples (7 cows x 2 time points: Day1 and Day4). "
            "The second row ('Label') encodes group: negative values for Day1, positive for Day4. "
            "This is a paired design (same cow measured at two time points). Some values are missing (empty cells). "
            "Please follow these steps in order: "
            "(1) Create an analysis plan named 'Bovine Urine Time Series' using create_plan tool (title field required). "
            "(2) Use read_file to read first 20 lines. "
            "(3) Use parse_omics_data to detect format. "
            "(4) Use update_plan_status to mark steps done. "
            "(5) Use the run_python tool to write and execute Python code that reads the CSV with pandas, "
            "handles missing values, separates Day1 and Day4 columns, computes fold change (Day4/Day1) "
            "for each compound, and prints the top 5 compounds with largest absolute fold change. "
            "Print the results clearly. "
            "(6) After the code runs, use read_file to check stdout.log in the run directory. "
            "(7) Based on the code output, summarize: dimensions, compounds, sample groups, "
            "extent of missing data, compounds with largest Day1-to-Day4 changes."
        )
        print(f"\n  Query: {query[:200]}...")

        t0 = time.perf_counter()
        response = await agent.chat(query)
        elapsed = time.perf_counter() - t0

        text = _extract_text(response)
        print(f"\n  proj04 latency: {elapsed:.0f}s")
        print(f"  Response length: {len(text)} chars")

        _save_audit_trail(agent, "proj04_bovine_timeseries", text)
        await agent.shutdown()

        assert response.role == "agent"
        assert len(text) > 100
        out = RESULTS_BASE / "proj04_bovine_timeseries"
        assert (out / "agent_response.md").exists()
        assert (out / "wire.jsonl").exists()
        # Verify code execution
        code_events = _count_code_exec_events(out / "wire.jsonl")
        assert code_events["run_python"] + code_events["run_r"] >= 1, \
            f"No code execution tool calls found: {code_events}"
        run_dirs = _find_run_dirs(Path(agent.runtime.config.output_dir))
        assert len(run_dirs) >= 1, f"No run directories found"
        for rd in run_dirs:
            findings = _verify_run_dir(rd, expected_script_ext="py")
            assert findings["script_exists"], f"script.py missing in {rd}"
            assert findings["stderr_exists"], f"stderr.log missing in {rd}"

    # ── proj05: Code Error Recovery ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_proj05_code_error_recovery(self):
        """Agent runs buggy Python code, inspects stderr.log, fixes it, re-runs successfully."""
        _skip_if_not_configured("anthropic")

        data_file = DATA_BASE / "proj01_human_cachexia_urine_nmr" / "metaboanalyst_human_cachexia.csv"
        assert data_file.exists(), f"Data file not found: {data_file}"

        runtime = _make_runtime("proj05_error_recovery")
        runtime.session.create_session(domain="metabolomics")
        agent = PassiAgent(runtime)
        await agent.initialize()

        query = (
            f"The data file is at {data_file}. First read the first 5 lines to understand columns. "
            "Then: (1) Write a short Python script using run_python tool that intentionally has a bug, "
            "such as trying to divide by zero or using an undefined variable. Run it. "
            "(2) When the execution fails (success=false), use read_file to read the stderr.log "
            "file from the run directory (the run_dir path is in the tool result) to see the full error details. "
            "(3) Fix the bug in the Python code and re-run it using run_python. "
            "The second execution should succeed (success=true) and print 'Bug fixed successfully!' "
            "(4) Summarize: what the original error was, and how you fixed it."
        )
        print(f"\n  Query: {query[:200]}...")

        t0 = time.perf_counter()
        response = await agent.chat(query)
        elapsed = time.perf_counter() - t0

        text = _extract_text(response)
        print(f"\n  proj05 latency: {elapsed:.0f}s")
        print(f"  Response length: {len(text)} chars")

        _save_audit_trail(agent, "proj05_error_recovery", text)
        await agent.shutdown()

        assert response.role == "agent"
        assert len(text) > 100, "Response too short — agent may have failed"

        out = RESULTS_BASE / "proj05_error_recovery"
        assert (out / "agent_response.md").exists()
        assert (out / "wire.jsonl").exists()

        # Verify at least 2 run directories (one failed, one successful)
        run_dirs = _find_run_dirs(Path(agent.runtime.config.output_dir))
        assert len(run_dirs) >= 2, \
            f"Expected >=2 run dirs (failed + successful), got {len(run_dirs)}"

        # Each should have script.py, stderr.log, metadata
        for rd in run_dirs:
            findings = _verify_run_dir(rd, expected_script_ext="py")
            assert findings["script_exists"], f"script.py missing in {rd}"
            assert findings["stderr_exists"], f"stderr.log missing in {rd}"

        # At least one success and one error
        has_success = False
        has_error = False
        for rd in run_dirs:
            meta_path = rd / "run_metadata.json"
            if meta_path.exists():
                md = json.loads(meta_path.read_text(encoding="utf-8"))
                if md.get("exit_code") == 0:
                    has_success = True
                else:
                    has_error = True
        assert has_success, "No successful run found (exit_code=0)"
        assert has_error, "No failed run found (non-zero exit_code)"

        # Verify wire.jsonl has at least 2 run_python calls
        code_events = _count_code_exec_events(out / "wire.jsonl")
        assert code_events["run_python"] >= 2, \
            f"Expected >=2 run_python calls, got {code_events}"


# ═══════════════════════════════════════════════════════════════════════
# E2E Summary Generator (run after all tests)
# ═══════════════════════════════════════════════════════════════════════


def _generate_summary() -> None:
    """Generate e2e_summary.md aggregating results across all projects."""
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    projects = [
        "proj01_human_cachexia", "proj02_human_kidney",
        "proj03_bovine_diet", "proj04_bovine_timeseries",
        "proj05_error_recovery",
    ]
    lines = [
        "# Metabolomics E2E Test Summary",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Project | Response | Wire Events | Provenance | Tasks | Plan | Run Dirs |",
        "|---------|----------|-------------|------------|-------|------|----------|",
    ]
    for proj in projects:
        d = RESULTS_BASE / proj
        if not d.exists():
            lines.append(f"| {proj} | MISSING | - | - | - | - | - |")
            continue
        resp = "OK" if (d / "agent_response.md").exists() else "MISSING"
        wire_count = _count_jsonl(d / "wire.jsonl")
        prov_count = _count_jsonl(d / "provenance.jsonl")
        task_count = _count_jsonl(d / "tasks.jsonl")
        plan = "OK" if (d / "plan.yaml").exists() else "MISSING"
        output_dir = d / "output"
        run_dirs = _find_run_dirs(output_dir) if output_dir.exists() else []
        run_count = len(run_dirs)
        lines.append(f"| {proj} | {resp} ({_file_size(d / 'agent_response.md')}) | "
                     f"{wire_count} | {prov_count} | {task_count} | {plan} | {run_count} |")
    (RESULTS_BASE / "e2e_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def _file_size(path: Path) -> str:
    if not path.exists():
        return "0B"
    size = path.stat().st_size
    if size > 1024:
        return f"{size / 1024:.0f}KB"
    return f"{size}B"


# Allow running summary standalone
if __name__ == "__main__":
    _generate_summary()
    print(f"Summary written to {RESULTS_BASE / 'e2e_summary.md'}")
