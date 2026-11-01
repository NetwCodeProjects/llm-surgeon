"""Step 7: scripts/acceptance_record.py on a mocked acceptance run (docs/acceptance.md, fake runtimes)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from conftest import ROOT

SCRIPT = ROOT / "scripts" / "acceptance_record.py"


def load_script():
    spec = importlib.util.spec_from_file_location("acceptance_record", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runs(env) -> set[str]:
    return {p.name for p in (env.workspace / "runs").iterdir()}


def test_record_of_a_mocked_acceptance_run(env):
    env.provision()  # the fake components exist already, so the first setup reuses them (clean setup fails below)
    code, out = env.run("--setup-only")
    assert code == 0, out
    (setup,) = runs(env)
    code, out = env.run("--setup-only", "--new-run")
    assert code == 0, out
    (reuse,) = runs(env) - {setup}
    # The search is stopped by its 1-minute budget; the pipeline is then killed after Heretic saved the export.
    code = env.run_killable("--new-run", "--max-runtime-minutes", "1", FAKE_TRIAL_SECONDS="8", FAKE_KILL_PARENT_AT="export-saved")
    assert code != 0
    (main,) = runs(env) - {setup, reuse}
    code, out = env.run("--resume", main)
    assert code == 0, out

    out_dir = env.tmp / "record"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--workspace", str(env.workspace), "--run", main, "--setup-run", setup, "--reuse-run", reuse,
         "--expect-model", "Qwen/Qwen3.8-27B", "--out", str(out_dir)],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert proc.returncode == 1, proc.stdout + proc.stderr  # not accepted: see clean_setup and code_revision
    record = json.loads((out_dir / "acceptance-record.json").read_text(encoding="utf-8"))
    results = {c["id"]: c["result"] for c in record["checks"]}
    assert results == {
        "clean_setup": "failed",
        "tool_reuse": "passed",
        "full_size_transform": "passed",
        "budget_export_only": "passed",
        "resume_after_interrupt": "passed",
        "promotion": "passed",
        "export_loading": "passed",
        "comparison": "passed",
        "code_revision": "unresolved",
    }, {c["id"]: c["detail"] for c in record["checks"]}
    by_id = {c["id"]: c for c in record["checks"]}
    assert by_id["clean_setup"]["detail"].startswith("not every component installed")
    assert by_id["resume_after_interrupt"]["values"]["stages"] == ["05"]
    assert record["verdict"] == "not accepted" and record["unresolved"] == ["code_revision"] and record["failed"] == ["clean_setup"]
    m = record["measurements"]
    assert set(m["artifact_bytes"]) == {"source", "master", "export:fp8", "export:gguf:iq4_xs", "export:gguf:iq3_s"}
    assert m["stages"]["05"]["outcome"] == "succeeded" and m["setup_seconds"] is not None
    md = (out_dir / "acceptance-record.md").read_text(encoding="utf-8")
    assert "# Acceptance record: not accepted" in md and "| Budget stop, then export-only with zero new trials | **passed** |" in md


def test_a_clean_setup_and_a_wrong_model_are_judged(tmp_path):
    module = load_script()
    run = tmp_path / "ws" / "runs" / "r1"
    (run / "receipts").mkdir(parents=True)
    receipt = {
        "stage_id": "01",
        "outcome": "succeeded",
        "duration_s": 285.2,
        "diagnostics": {"component_actions": {"heretic": "installed", "vllm": "installed"}},
    }
    (run / "receipts" / "01-environment.json").write_text(json.dumps(receipt))
    (run / "configs").mkdir()
    (run / "configs" / "resolved-config.json").write_text(json.dumps({"config": {"model": {"url": "Qwen/Qwen3.5-2B"}}}))
    (run / "receipts" / "05-abliterate.json").write_text(json.dumps({"stage_id": "05", "outcome": "succeeded", "duration_s": 1.0}))
    view = module.RunView(tmp_path / "ws", "r1")
    clean = module.setup_check(view, "clean_setup", "t", "installed")
    assert clean["result"] == "passed" and clean["detail"] == "all 2 components installed in 285.2 s"
    assert module.setup_check(view, "tool_reuse", "t", "reused")["result"] == "failed"
    assert module.setup_check(None, "tool_reuse", "t", "reused")["result"] == "unresolved"
    wrong = module.full_size_check(view, "Qwen/Qwen3.8-27B")
    assert wrong["result"] == "failed" and wrong["detail"] == "the run used Qwen/Qwen3.5-2B, not Qwen/Qwen3.8-27B"
