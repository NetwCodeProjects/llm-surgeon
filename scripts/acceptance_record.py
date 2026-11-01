#!/usr/bin/env python3
"""Turn a finished acceptance run into acceptance-record.json and acceptance-record.md (docs/acceptance.md).

Stdlib only: it reads what the pipeline already wrote (receipts, receipt history, events, study history,
metrics, manifests) and never runs a tool. Every check is ``passed``, ``failed`` or ``unresolved`` (the
evidence is missing); the record is ``accepted`` only when every check passed. Paths recorded on the GPU
host (e.g. /root/ws/...) are rebased onto ``--workspace`` so the record can be rebuilt from a copy.

    python3 scripts/acceptance_record.py --workspace WS --setup-run A --reuse-run B --run C
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECORD_SCHEMA = "1.0"
STAGE_LINE = re.compile(r"^\[(\d\d) [a-z_]+\] (starting|succeeded|partial|failed|interrupted|blocked|reused)\b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check(check_id: str, title: str, result: str, detail: str, evidence: list[str] | None = None, **values: Any) -> dict[str, Any]:
    return {"id": check_id, "title": title, "result": result, "detail": detail, "evidence": evidence or [], **({"values": values} if values else {})}


class RunView:
    """One run directory, with host paths from its receipts rebased onto the local workspace."""

    def __init__(self, workspace: Path, run_id: str) -> None:
        self.workspace = workspace
        self.run_id = run_id
        self.dir = workspace / "runs" / run_id
        if not self.dir.is_dir():
            raise SystemExit(f"run {run_id} not found under {workspace / 'runs'}")
        resolved = self.dir / "configs" / "resolved-config.json"
        self.config = read_json(resolved)["config"] if resolved.is_file() else {}
        self.host_root = (self.config.get("storage") or {}).get("root")

    def local(self, recorded: str | None) -> Path | None:
        if not recorded:
            return None
        path = Path(recorded)
        if path.exists():
            return path
        if self.host_root:
            try:
                return self.workspace / Path(recorded).relative_to(self.host_root)
            except ValueError:
                pass
        return path

    def receipt(self, stage: str) -> dict[str, Any] | None:
        found = sorted((self.dir / "receipts").glob(f"{stage}-*.json"))
        return read_json(found[0]) if found else None

    def metric(self, name: str) -> Any:
        path = self.dir / "metrics" / name
        return read_json(path) if path.is_file() else None

    def events(self) -> list[dict[str, Any]]:
        path = self.dir / "logs" / "events.jsonl"
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # a line cut short by a killed process
        return rows


# ------------------------------------------------------------------------------------------------ checks
def setup_check(view: RunView | None, check_id: str, title: str, want: str) -> dict[str, Any]:
    if view is None:
        return check(check_id, title, "unresolved", "run not given")
    receipt = view.receipt("01")
    if receipt is None:
        return check(check_id, title, "unresolved", f"run {view.run_id} has no stage 01 receipt")
    actions = (receipt.get("diagnostics") or {}).get("component_actions") or {}
    evidence = [str(view.dir / "receipts" / "01-environment.json")]
    if receipt.get("outcome") != "succeeded":
        return check(check_id, title, "failed", f"stage 01 {receipt.get('outcome')}", evidence, actions=actions)
    others = {c: a for c, a in actions.items() if a != want}
    ok = bool(actions) and not others
    detail = f"all {len(actions)} components {want} in {receipt.get('duration_s')} s" if ok else f"not every component {want}: {others or 'none recorded'}"
    return check(check_id, title, "passed" if ok else "failed", detail, evidence, actions=actions, duration_s=receipt.get("duration_s"))


def full_size_check(view: RunView, expect_model: str | None) -> dict[str, Any]:
    title = "Full-size model loaded and transformed (stage 05)"
    receipt = view.receipt("05")
    if receipt is None:
        return check("full_size_transform", title, "unresolved", "no stage 05 receipt")
    candidate = view.metric("candidate.json") or {}
    params = sum(((candidate.get("dtype_summary") or {}).get("dtype_param_counts") or {}).values()) or None
    source = view.metric("quantization.json")
    model = (view.config.get("model") or {}).get("url")
    ok = receipt.get("outcome") == "succeeded" and (expect_model is None or model == expect_model)
    if expect_model is not None and model != expect_model:
        return check("full_size_transform", title, "failed", f"the run used {model}, not {expect_model}", model=model)
    detail = f"{model}: stage 05 {receipt.get('outcome')} in {receipt.get('duration_s')} s" + (
        f", {params / 1e9:.2f}B parameters in the candidate" if params else ""
    )
    return check(
        "full_size_transform",
        title,
        "passed" if ok else "failed",
        detail,
        [str(view.dir / "receipts" / "05-abliterate.json")],
        model=model,
        candidate_parameters=params,
        input_bytes=sum(f["size_bytes"] for f in (source or {}).get("input", {}).get("files", [])) or None,
    )


def study_history(view: RunView) -> tuple[list[dict[str, Any]], Path | None]:
    receipt = view.receipt("05") or {}
    path = view.local((receipt.get("diagnostics") or {}).get("history"))
    if path is None or not path.is_file():
        return [], path
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], path


def budget_check(view: RunView) -> dict[str, Any]:
    title = "Budget stop, then export-only with zero new trials"
    entries, path = study_history(view)
    if not entries:
        return check("budget_export_only", title, "unresolved", f"no study history ({path})")
    stopped = [e for e in entries if e.get("phase") == "completion" and e.get("action") in ("new", "continue", "fork-continue") and e.get("budget_stopped")]
    exports = [e for e in entries if e.get("action") == "export" and e.get("phase") in ("intent", "completion")]
    evidence = [str(path)]
    if not stopped:
        return check("budget_export_only", title, "failed", "the search finished within its budget (no budget_stopped completion)", evidence)
    if not exports or exports[-1].get("phase") != "completion":
        return check("budget_export_only", title, "failed", "no completed export after the budget stop", evidence)
    intent = next((e for e in reversed(exports[:-1]) if e.get("phase") == "intent"), None)
    before = (intent or {}).get("before") or {}
    after = exports[-1].get("after") or {}
    same = before.get("trial_set_sha256") is not None and before.get("trial_set_sha256") == after.get("trial_set_sha256")
    trials = after.get("trials")
    detail = (
        f"search stopped by the budget after {stopped[-1].get('after', {}).get('trials')} trials; export restored trial "
        f"{(exports[-1].get('export') or {}).get('restored_trial')} and the study kept {trials} trials"
        if same
        else f"the export changed the study's trials ({before.get('trials')} -> {trials})"
    )
    return check("budget_export_only", title, "passed" if same else "failed", detail, evidence, trials=trials)


def resume_check(view: RunView) -> dict[str, Any]:
    title = "Resume after an interrupted stage"
    attempts: dict[str, list[str]] = {}
    for event in view.events():
        match = STAGE_LINE.match(event.get("message") or "")
        if match:
            attempts.setdefault(match.group(1), []).append(match.group(2))
    resumed = {}
    for stage, steps in attempts.items():
        starts = steps.count("starting")
        final = view.receipt(stage) or {}
        # an attempt that did not finish (killed) or ended interrupted, followed by a successful one
        unfinished = starts > sum(1 for s in steps if s != "starting" and s != "reused") or "interrupted" in steps
        if starts >= 2 and unfinished and final.get("outcome") == "succeeded":
            resumed[stage] = steps
    history = sorted(p.name for p in (view.dir / "receipts" / "history").glob("*")) if (view.dir / "receipts" / "history").is_dir() else []
    evidence = [str(view.dir / "logs" / "events.jsonl")] + [str(view.dir / "receipts" / "history" / h) for h in history]
    if not attempts:
        return check("resume_after_interrupt", title, "unresolved", "no events.jsonl")
    if not resumed:
        return check("resume_after_interrupt", title, "failed", "no stage was interrupted and then completed in this run", evidence)
    detail = "; ".join(f"stage {s}: {' -> '.join(steps)}" for s, steps in sorted(resumed.items()))
    return check("resume_after_interrupt", title, "passed", detail, evidence, stages=sorted(resumed))


def promotion_check(view: RunView) -> dict[str, Any]:
    title = "Decision-table promotion (stage 07)"
    receipt = view.receipt("07")
    if receipt is None:
        return check("promotion", title, "unresolved", "no stage 07 receipt")
    promotion = (receipt.get("diagnostics") or {}).get("promotion")
    ok = receipt.get("outcome") == "succeeded" and promotion == "promoted:qualified"
    return check(
        "promotion",
        title,
        "passed" if ok else "failed",
        f"stage 07 {receipt.get('outcome')}, master {promotion}",
        [str(view.dir / "receipts" / "07-select_master.json"), str(view.dir / "metrics" / "selection.json")],
        promotion=promotion,
    )


def loading_check(view: RunView) -> dict[str, Any]:
    title = "Every export loads in its runtime (stage 09)"
    verification = view.metric("verification.json")
    if verification is None:
        return check("export_loading", title, "unresolved", "no metrics/verification.json")
    rows, bad = {}, []
    for role, item in sorted(verification.items()):
        if not role.startswith("export:"):
            continue
        smoke = item.get("smoke") or {}
        rows[role] = {
            "backend": smoke.get("backend"),
            "passed": smoke.get("passed"),
            "representative": smoke.get("representative"),
            "load_seconds": smoke.get("load_seconds"),
        }
        if not smoke.get("ran") or not smoke.get("passed") or smoke.get("representative") is False or item.get("integrity_problems"):
            bad.append(role)
    if not rows:
        return check("export_loading", title, "unresolved", "no exports were verified")
    detail = f"{len(rows) - len(bad)}/{len(rows)} exports loaded and generated" + (f"; not passed: {bad}" if bad else "")
    return check("export_loading", title, "failed" if bad else "passed", detail, [str(view.dir / "metrics" / "verification.json")], exports=rows)


def comparison_check(view: RunView) -> dict[str, Any]:
    title = "Master/export comparison (stage 09)"
    doc = view.metric("comparison.json")
    if doc is None:
        return check("comparison", title, "unresolved", "no metrics/comparison.json (comparison disabled or no exports)")
    outcomes = {role: e.get("outcome") for role, e in doc.get("exports", {}).items()}
    base = doc.get("baseline") or {}
    ok = bool(base.get("measured")) and bool(outcomes) and all(o == "passed" for o in outcomes.values())
    detail = f"baseline {base.get('role')} measured={base.get('measured')}; " + ", ".join(f"{r} {o}" for r, o in sorted(outcomes.items()))
    return check("comparison", title, "passed" if ok else "failed", detail, [str(view.dir / "metrics" / "comparison.json")], outcomes=outcomes)


# ------------------------------------------------------------------------------------------ measurements
def tree_bytes(root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if (st.st_dev, st.st_ino) in seen:
                continue  # hard links (master files linked from the candidate) count once
            seen.add((st.st_dev, st.st_ino))
            total += st.st_size
    return total


def measurements(view: RunView, setup: RunView | None, reuse: RunView | None) -> dict[str, Any]:
    stages = {}
    for path in sorted((view.dir / "receipts").glob("[0-9][0-9]-*.json")):
        r = read_json(path)
        res = r.get("resources") or {}
        stages[r["stage_id"]] = {
            "name": r.get("stage_name"),
            "outcome": r.get("outcome"),
            "duration_s": r.get("duration_s"),
            "peak_gpu_memory_mib": res.get("peak_gpu_memory_mib"),
            "peak_gpu_utilization_pct": res.get("peak_gpu_utilization_pct"),
            "peak_host_memory_bytes": res.get("peak_host_memory_bytes"),
            "min_disk_free_bytes": res.get("min_disk_free_bytes"),
        }

    def peak(key: str) -> Any:
        values = [s[key] for s in stages.values() if s.get(key) is not None]
        return max(values) if values else None

    artifacts = {}
    index = view.dir / "artifacts.json"
    for role, entry in (read_json(index)["artifacts"] if index.is_file() else {}).items():
        manifest_path = view.local(entry.get("manifest"))
        if not (manifest_path and manifest_path.is_file()):
            manifest_path = view.workspace / "manifests" / f"{entry['artifact_id']}.json"  # the global copy
        if manifest_path.is_file():
            files = read_json(manifest_path)["artifact"]["files"]
            artifacts[role] = sum(f["size_bytes"] for f in files)
    disk = {name: tree_bytes(view.workspace / name) for name in ("exports", "models", "cache", "tools", "runs") if (view.workspace / name).is_dir()}
    return {
        "stage_seconds": sum(s["duration_s"] or 0 for s in stages.values()),
        "setup_seconds": (setup.receipt("01") or {}).get("duration_s") if setup else None,
        "reuse_setup_seconds": (reuse.receipt("01") or {}).get("duration_s") if reuse else None,
        "peak_gpu_memory_mib": peak("peak_gpu_memory_mib"),
        "peak_host_memory_bytes": peak("peak_host_memory_bytes"),  # MemTotal - MemAvailable: the whole host in a container
        "min_disk_free_bytes": min((s["min_disk_free_bytes"] for s in stages.values() if s.get("min_disk_free_bytes") is not None), default=None),
        "stages": stages,
        "artifact_bytes": artifacts,
        "workspace_bytes": disk,
        "note": "Resource values are samples (logging.resource_sample_seconds); peaks between samples may be missed. Host memory is "
        "MemTotal - MemAvailable from /proc/meminfo, which in a container covers the whole host, not only this run.",
    }


def provenance(view: RunView) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preflight = view.metric("preflight.json") or {}
    summary_path = view.dir / "reports" / "run-summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    components = view.metric("components.json") or {}
    code = preflight.get("code") or {}
    extra: list[dict[str, Any]] = []
    if not code.get("git_commit"):
        extra.append(
            check(
                "code_revision",
                "Code revision recorded by the run",
                "unresolved",
                "preflight did not record a git commit (not a git checkout, or a build older than the record)",
            )
        )
    elif code.get("git_dirty"):
        extra.append(check("code_revision", "Code revision recorded by the run", "failed", f"{code['git_commit']} with uncommitted changes to tracked files"))
    else:
        extra.append(check("code_revision", "Code revision recorded by the run", "passed", code["git_commit"]))
    return {
        "platform_version": code.get("version") or summary.get("platform_version"),
        "git_commit": code.get("git_commit"),
        "git_dirty": code.get("git_dirty"),
        "config_hash": summary.get("config_hash") or (read_json(view.dir / "status.json") if (view.dir / "status.json").is_file() else {}).get("config_hash"),
        "config_profiles": summary.get("config_profiles") or [],
        "model": view.config.get("model"),
        "gpus": [{k: g.get(k) for k in ("name", "compute_capability", "memory_total_mib", "driver_version")} for g in preflight.get("gpus") or []],
        "cuda_driver_version": preflight.get("cuda_driver_version"),
        "nvcc_version": preflight.get("nvcc_version"),
        "available_cpus": preflight.get("available_cpus"),
        "tools": {
            name: {"fingerprint": c.get("fingerprint"), "versions": c.get("versions"), "build": c.get("build")} for name, c in sorted(components.items())
        },
    }, extra


def record_tool_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def build_record(workspace: Path, run_id: str, setup_id: str | None, reuse_id: str | None, expect_model: str | None = None) -> dict[str, Any]:
    view = RunView(workspace, run_id)
    setup = RunView(workspace, setup_id) if setup_id else None
    reuse = RunView(workspace, reuse_id) if reuse_id else None
    prov, extra = provenance(view)
    checks = [
        setup_check(setup, "clean_setup", "Clean setup installs every component", "installed"),
        setup_check(reuse, "tool_reuse", "A second setup reuses every component", "reused"),
        full_size_check(view, expect_model),
        budget_check(view),
        resume_check(view),
        promotion_check(view),
        loading_check(view),
        comparison_check(view),
        *extra,
    ]
    results = [c["result"] for c in checks]
    return {
        "schema_version": RECORD_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdict": "accepted" if all(r == "passed" for r in results) else "not accepted",
        "runs": {"acceptance": run_id, "setup": setup_id, "reuse_setup": reuse_id},
        "expected_model": expect_model,
        "provenance": prov,
        "record_tool_commit": record_tool_commit(),
        "checks": checks,
        "unresolved": [c["id"] for c in checks if c["result"] == "unresolved"],
        "failed": [c["id"] for c in checks if c["result"] == "failed"],
        "measurements": measurements(view, setup, reuse),
    }


# ------------------------------------------------------------------------------------------------ output
def secs(n: float | None) -> str:
    return "n/a" if n is None else f"{n:.0f} s"


def human(n: float | None, unit: str = "B") -> str:
    if n is None:
        return "n/a"
    if unit == "MiB":
        return f"{n / 1024:.1f} GiB"
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or suffix == "TiB":
            return f"{n:.1f} {suffix}" if suffix != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"


def markdown(record: dict[str, Any]) -> str:
    p, m = record["provenance"], record["measurements"]
    gpu = ", ".join(f"{g['name']} (cc {g['compute_capability']})" for g in p["gpus"]) or "n/a"
    lines = [
        f"# Acceptance record: {record['verdict']}",
        "",
        f"Run `{record['runs']['acceptance']}` (setup `{record['runs']['setup'] or 'not given'}`, reuse `{record['runs']['reuse_setup'] or 'not given'}`), "
        f"generated {record['generated_utc']}.",
        "",
        "| | |",
        "|---|---|",
        f"| Code | llm-surgeon {p['platform_version']} at `{p['git_commit'] or 'commit not recorded'}`{' (dirty)' if p['git_dirty'] else ''} |",
        f"| Model | {(p['model'] or {}).get('url')} @ `{(p['model'] or {}).get('revision')}` |",
        f"| GPU | {gpu}; driver CUDA {p['cuda_driver_version']}; nvcc {p['nvcc_version']}; usable CPUs {p['available_cpus'] or 'n/a'} |",
        f"| Config hash | `{p['config_hash']}` |",
        *[f"| Profile | `{Path(x['path']).name}` `{x['sha256'][:16]}` |" for x in p["config_profiles"]],
        *[f"| {name} | {json.dumps(t['versions'], sort_keys=True)} |" for name, t in p["tools"].items()],
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
        *[f"| {c['title']} | **{c['result']}** | {c['detail']} |" for c in record["checks"]],
        "",
    ]
    if record["unresolved"]:
        lines += ["Unresolved (evidence missing): " + ", ".join(record["unresolved"]), ""]
    lines += [
        "## Measurements",
        "",
        f"Stages {secs(m['stage_seconds'])}; setup {secs(m['setup_seconds'])}, setup reuse {secs(m['reuse_setup_seconds'])}; "
        f"peak GPU memory {human(m['peak_gpu_memory_mib'], 'MiB')}; peak host memory in use {human(m['peak_host_memory_bytes'])}; "
        f"lowest free disk {human(m['min_disk_free_bytes'])}.",
        "",
        "| Stage | Outcome | Seconds | Peak GPU memory | Peak GPU util % | Peak host memory in use |",
        "|---|---|---:|---:|---:|---:|",
        *[
            f"| {sid} {s['name']} | {s['outcome']} | {secs(s['duration_s'])} | {human(s['peak_gpu_memory_mib'], 'MiB')} | "
            f"{s['peak_gpu_utilization_pct'] if s['peak_gpu_utilization_pct'] is not None else 'n/a'} | {human(s['peak_host_memory_bytes'])} |"
            for sid, s in sorted(m["stages"].items())
        ],
        "",
        "| Artifact | Size |",
        "|---|---:|",
        *[f"| {role} | {human(size)} |" for role, size in sorted(m["artifact_bytes"].items())],
        "",
        "| Workspace directory | Size on disk (hard links once) |",
        "|---|---:|",
        *[f"| {name}/ | {human(size)} |" for name, size in m["workspace_bytes"].items()],
        "",
        m["note"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--run", required=True, help="the acceptance run (budget stop, interruption, resume)")
    parser.add_argument("--setup-run", help="the first --setup-only run on a clean workspace")
    parser.add_argument("--reuse-run", help="the second --setup-only run")
    parser.add_argument("--expect-model", help="the full-size model the run must have used, e.g. Qwen/Qwen3.8-27B")
    parser.add_argument("--out", type=Path, help="output directory (default: runs/<run>/reports)")
    args = parser.parse_args(argv)
    record = build_record(args.workspace, args.run, args.setup_run, args.reuse_run, args.expect_model)
    out = args.out or args.workspace / "runs" / args.run / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "acceptance-record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "acceptance-record.md").write_text(markdown(record), encoding="utf-8")
    print(f"{record['verdict']}: {out / 'acceptance-record.md'}")
    for c in record["checks"]:
        print(f"  {c['result']:<10} {c['id']}: {c['detail']}")
    return 0 if record["verdict"] == "accepted" else 1


if __name__ == "__main__":
    sys.exit(main())
